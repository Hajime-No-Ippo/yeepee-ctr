"""Step 2 features (train-only global tables; each target row uses only its own history):
 repeat-click: decayed own-history repeat counts, user repeat propensity, item/user CTR conditional on candidate-in-history (train impressions < C)
 directional time-decayed co-visitation a->b from the train click log (< C) + impression co-vis CTR (history item a -> candidate b, train rows < C)
Usage: python rcov.py NAME... -> feat_rc/NAME.parquet"""
import sys, os, time, numpy as np, polars as pl
sys.path.insert(0, '/workspace/huawei/v3')
from features import load, click_log
from split import J20, DAY, first_events
OUT = '/workspace/huawei/v8/feat_cv'
TE, TR, EE, ER, NI, NU = load('/workspace/huawei/v3/data')
SPEC = {**{f'f{k}': (J20 + k * DAY, DAY) for k in range(1, 7)}, 'val25': (J20 + 5 * DAY, None), 'val26': (J20 + 6 * DAY, None), 'eval': (J20 + 7 * DAY, 'eval'),
        **{f'e{k}': (J20 + k * DAY, (5 - k) * DAY) for k in range(1, 5)}, **{f'E{k}': (J20 + k * DAY, (7 - k) * DAY) for k in range(1, 7)},
        **{f'd{k}': (J20 + k * DAY, (6 - k) * DAY) for k in range(1, 6)}}
H = 3600
def build(C, T_ev, T_rows):
    P_ev = TE.filter(pl.col('ts') < C)
    P_rows = TR.join(P_ev.select('eid', 'u', 'ts', 'h', 'h_ts'), on='eid', how='inner')
    g = P_rows['clicked'].mean()
    CLg = click_log(C, P_ev, P_rows.select('eid', 'u', 'ii', 'clicked', 'ts'), T_ev.head(0))  # train only
    # ---- directional co-visitation a->b (b clicked after a, within 24h, up to 15 steps) ----
    s = CLg.sort('u', 'ts')
    parts = []
    for lag in range(1, 31):
        parts.append(s.with_columns(pl.col('ii').shift(lag).over('u').alias('a'), pl.col('ts').shift(lag).over('u').alias('ta'))
                      .drop_nulls('a').filter((pl.col('ts') - pl.col('ta') <= 3 * DAY) & (pl.col('a') != pl.col('ii')))
                      .select('a', pl.col('ii').alias('b'),
                              (-(pl.col('ts') - pl.col('ta')) / DAY).exp().alias('wg'),
                              ((-(pl.col('ts') - pl.col('ta')) / DAY) - (C - pl.col('ts')) / (3 * DAY)).exp().alias('wr')))
    cv = pl.concat(parts).group_by('a', 'b').agg(pl.col('wg').sum(), pl.col('wr').sum())
    out = cv.group_by('a').agg(pl.col('wg').sum().alias('og'), pl.col('wr').sum().alias('or'))
    cv = cv.join(out, on='a').select('a', 'b', (pl.col('wg') / (pl.col('og') + 1)).cast(pl.Float32).alias('cvg'),
                                     (pl.col('wr') / (pl.col('or') + 0.3)).cast(pl.Float32).alias('cvr'), pl.col('wg').cast(pl.Float32).alias('cvn'))
    del parts, s
    R = T_rows.select('rowid', 'eid', 'ii').join(T_ev.select('eid', 'ts', 'h', 'h_ts'), on='eid', how='left').with_row_index('r')
    X = (R.select('r', 'ii', 'ts', pl.col('h').list.tail(20), pl.col('h_ts').list.tail(20))
           .with_columns(pl.int_ranges(pl.col('h').list.len(), 0, -1).alias('pos')).explode(['h', 'h_ts', 'pos']).drop_nulls('h')
           .with_columns(((pl.col('ts') - pl.col('h_ts')) / DAY).alias('ag'), (0.85 ** (pl.col('pos') - 1)).alias('pw')))
    Xc = (X.rename({'h': 'a', 'ii': 'b'}).join(cv, on=['a', 'b'], how='left')
            .join(cv.select(pl.col('a').alias('b'), pl.col('b').alias('a'), pl.col('cvg').alias('cvb')), on=['a', 'b'], how='left').fill_null(0)
            .with_columns((-pl.col('ag') / 3).exp().alias('tw')))
    A = Xc.group_by('r').agg((pl.col('cvg') * pl.col('pw')).sum().alias('g'), (pl.col('cvr') * pl.col('pw')).sum().alias('r_'),
        (pl.col('cvg') * pl.col('tw')).sum().alias('gt'), (pl.col('cvr') * pl.col('tw')).sum().alias('rt'), pl.col('cvg').max().alias('gmax'),
        pl.col('cvg').filter(pl.col('pos') <= 3).sum().alias('last3'), (pl.col('cvb') * pl.col('pw')).sum().alias('back'),
        pl.col('cvn').sum().alias('n'), (pl.col('cvg') > 0).sum().alias('nz'))
    R = R.join(A, on='r', how='left').sort('r')
    fc = ['g', 'r_', 'gt', 'rt', 'gmax', 'last3', 'back', 'n', 'nz']
    R = R.with_columns([pl.col(c).fill_null(0).cast(pl.Float32) for c in fc]).with_columns((pl.col('g') - pl.col('back')).alias('dir'))
    zc = ['g', 'r_', 'gt', 'rt', 'last3', 'dir', 'n']
    R = R.with_columns([((pl.col(c) - pl.col(c).mean().over('eid')) / (pl.col(c).std().over('eid') + 1e-6)).cast(pl.Float32).alias(c + '_z') for c in zc]
                       + [pl.col(c).rank('average').over('eid').cast(pl.Float32).alias(c + '_rk') for c in ('g', 'rt')])
    keep = fc + ['dir'] + [c + '_z' for c in zc] + ['g_rk', 'rt_rk']
    return R.select(['rowid'] + keep).rename({c: 'rc_cv2_' + c for c in keep})
for name in sys.argv[1:]:
    t0 = time.time(); C, W = SPEC[name]
    if os.path.exists(f'{OUT}/{name}.parquet'): continue
    if W == 'eval': R = build(C, EE, ER)
    else:
        T_ev = first_events(TE, C, W); R = build(C, T_ev, TR.join(T_ev.select('eid'), on='eid', how='semi'))
    R.write_parquet(f'{OUT}/{name}.parquet'); print(name, R.shape, f'{time.time()-t0:.0f}s', flush=True)
