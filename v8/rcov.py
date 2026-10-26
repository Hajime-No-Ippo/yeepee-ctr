"""Step 2 features (train-only global tables; each target row uses only its own history):
 repeat-click: decayed own-history repeat counts, user repeat propensity, item/user CTR conditional on candidate-in-history (train impressions < C)
 directional time-decayed co-visitation a->b from the train click log (< C) + impression co-vis CTR (history item a -> candidate b, train rows < C)
Usage: python rcov.py NAME... -> feat_rc/NAME.parquet"""
import sys, os, time, numpy as np, polars as pl
sys.path.insert(0, '/workspace/huawei/v3')
from features import load, click_log
from split import J20, DAY, first_events
OUT = '/workspace/huawei/v8/feat_rc'
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
    for lag in range(1, 16):
        parts.append(s.with_columns(pl.col('ii').shift(lag).over('u').alias('a'), pl.col('ts').shift(lag).over('u').alias('ta'))
                      .drop_nulls('a').filter((pl.col('ts') - pl.col('ta') <= 24 * H) & (pl.col('a') != pl.col('ii')))
                      .select('a', pl.col('ii').alias('b'),
                              (-(pl.col('ts') - pl.col('ta')) / (6 * H)).exp().alias('wg'),
                              ((-(pl.col('ts') - pl.col('ta')) / (6 * H)) - (C - pl.col('ts')) / (3 * DAY)).exp().alias('wr')))
    cv = pl.concat(parts).group_by('a', 'b').agg(pl.col('wg').sum(), pl.col('wr').sum())
    out = cv.group_by('a').agg(pl.col('wg').sum().alias('og'), pl.col('wr').sum().alias('or'))
    cv = cv.join(out, on='a').select('a', 'b', (pl.col('wg') / (pl.col('og') + 1)).cast(pl.Float32).alias('cvg'),
                                     (pl.col('wr') / (pl.col('or') + 0.3)).cast(pl.Float32).alias('cvr'), pl.col('wg').cast(pl.Float32).alias('cvn'))
    del parts, s
    # ---- impression co-vis CTR: history item a (last 5) -> candidate b, train rows < C ----
    ict = P_rows.group_by('ii').agg(((pl.col('clicked').sum() + 20 * g) / (pl.len() + 20)).alias('ictr'))
    IP = (P_rows.select('ii', 'clicked', pl.col('h').list.tail(5).alias('a')).explode('a').drop_nulls('a')
            .group_by('a', 'ii').agg(pl.len().alias('n'), pl.col('clicked').sum().alias('c')))
    IP = IP.join(ict, on='ii', how='left').select('a', pl.col('ii').alias('b'), pl.col('n').cast(pl.Float32).alias('icn'),
                 ((pl.col('c') + 20 * pl.col('ictr')) / (pl.col('n') + 20) / pl.col('ictr')).cast(pl.Float32).alias('iclift'))
    # ---- repeat CTR tables (train impressions < C) ----
    PR = P_rows.select('u', 'ii', 'clicked', pl.col('h').list.contains(pl.col('ii')).alias('inh'))
    rg = PR.group_by('inh').agg(pl.col('clicked').mean().alias('m')); gm = dict(zip(rg['inh'].to_list(), rg['m'].to_list()))
    irep = PR.group_by('ii', 'inh').agg(pl.len().alias('n'), pl.col('clicked').sum().alias('c')).with_columns(
        ((pl.col('c') + 20 * pl.when(pl.col('inh')).then(gm.get(True, g)).otherwise(gm.get(False, g))) / (pl.col('n') + 20)).cast(pl.Float32).alias('i_ctr_inh'))
    urep = PR.group_by('u', 'inh').agg(pl.len().alias('n'), pl.col('clicked').sum().alias('c')).with_columns(
        ((pl.col('c') + 10 * pl.when(pl.col('inh')).then(gm.get(True, g)).otherwise(gm.get(False, g))) / (pl.col('n') + 10)).cast(pl.Float32).alias('u_ctr_inh'))
    # ---- target rows ----
    R = T_rows.select('rowid', 'eid', 'ii').join(T_ev.select('eid', 'u', 'ts', 'h', 'h_ts'), on='eid', how='left').with_row_index('r')
    R = R.with_columns(pl.col('h').list.contains(pl.col('ii')).alias('inh'))
    R = R.join(irep.select('ii', 'inh', 'i_ctr_inh'), on=['ii', 'inh'], how='left').join(urep.select('u', 'inh', 'u_ctr_inh'), on=['u', 'inh'], how='left')
    R = R.with_columns(pl.col('i_ctr_inh').fill_null(pl.when(pl.col('inh')).then(gm.get(True, g)).otherwise(gm.get(False, g))),
                       pl.col('u_ctr_inh').fill_null(pl.when(pl.col('inh')).then(gm.get(True, g)).otherwise(gm.get(False, g))))
    # user repeat propensity from own history: share of history clicks on an item already clicked earlier in history
    E = T_ev.select('eid', pl.col('h').list.len().alias('hl'), (pl.col('h').list.len() - pl.col('h').list.n_unique()).alias('hrep'))
    R = R.join(E.with_columns((pl.col('hrep') / pl.col('hl').clip(1)).cast(pl.Float32).alias('rc_urep')).select('eid', 'rc_urep'), on='eid', how='left')
    X = (R.select('r', 'ii', 'ts', pl.col('h').list.tail(10), pl.col('h_ts').list.tail(10))
           .with_columns(pl.int_ranges(pl.col('h').list.len(), 0, -1).alias('pos')).explode(['h', 'h_ts', 'pos']).drop_nulls('h')
           .with_columns(((pl.col('ts') - pl.col('h_ts')) / DAY).alias('ag'), (0.8 ** (pl.col('pos') - 1)).alias('pw')))
    # repeat (full history, not just last 10)
    Xr = (R.select('r', 'ii', 'ts', 'h', 'h_ts').explode(['h', 'h_ts']).drop_nulls('h').filter(pl.col('h') == pl.col('ii'))
            .with_columns(((pl.col('ts') - pl.col('h_ts')) / DAY).alias('ag'))
            .group_by('r').agg((-pl.col('ag') / 0.25).exp().sum().alias('rc_dec6h'), (-pl.col('ag')).exp().sum().alias('rc_dec1d'),
                               (-pl.col('ag') / 7).exp().sum().alias('rc_dec7d'), pl.col('ag').min().alias('rc_age_min')))
    Xc = (X.rename({'h': 'a', 'ii': 'b'}).join(cv, on=['a', 'b'], how='left')
            .join(cv.select(pl.col('a').alias('b'), pl.col('b').alias('a'), pl.col('cvg').alias('cvb')), on=['a', 'b'], how='left')
            .join(IP, on=['a', 'b'], how='left').fill_null(0)
            .with_columns((pl.col('pw') * (-pl.col('ag') / 2).exp()).alias('tw')))
    A = Xc.group_by('r').agg(
        (pl.col('cvg') * pl.col('pw')).sum().alias('cv_g'), (pl.col('cvr') * pl.col('pw')).sum().alias('cv_r'),
        (pl.col('cvg') * pl.col('tw')).sum().alias('cv_gt'), pl.col('cvg').max().alias('cv_gmax'),
        pl.col('cvg').filter(pl.col('pos') == 1).sum().alias('cv_last'), pl.col('cvr').filter(pl.col('pos') == 1).sum().alias('cvr_last'),
        (pl.col('cvb') * pl.col('pw')).sum().alias('cv_back'), pl.col('cvn').sum().alias('cv_n'),
        ((pl.col('iclift') - 1).filter(pl.col('icn') > 0) * pl.col('pw').filter(pl.col('icn') > 0)).sum().alias('ic_lift_w'),
        pl.col('iclift').filter(pl.col('icn') > 0).max().alias('ic_lift_max'), pl.col('icn').sum().alias('ic_n'))
    R = R.join(A, on='r', how='left').join(Xr, on='r', how='left').sort('r')
    fc = ['cv_g', 'cv_r', 'cv_gt', 'cv_gmax', 'cv_last', 'cvr_last', 'cv_back', 'cv_n', 'ic_lift_w', 'ic_lift_max', 'ic_n', 'rc_dec6h', 'rc_dec1d', 'rc_dec7d']
    R = R.with_columns([pl.col(c).fill_null(0).cast(pl.Float32) for c in fc] + [pl.col('rc_age_min').cast(pl.Float32)])
    R = R.with_columns((pl.col('cv_g') - pl.col('cv_back')).alias('cv_dir'))
    zc = ['cv_g', 'cv_r', 'cv_gt', 'cv_last', 'cv_dir', 'ic_lift_w', 'i_ctr_inh', 'rc_dec1d']
    R = R.with_columns([((pl.col(c) - pl.col(c).mean().over('eid')) / (pl.col(c).std().over('eid') + 1e-6)).cast(pl.Float32).alias(c + '_z') for c in zc])
    R = R.with_columns(pl.col('inh').cast(pl.Int8))
    keep = ['rowid'] + fc + ['rc_age_min', 'cv_dir', 'i_ctr_inh', 'u_ctr_inh', 'rc_urep'] + [c + '_z' for c in zc]
    return R.select(keep).rename({c: 'rc_' + c for c in keep if c != 'rowid' and not c.startswith('rc_')})
for name in sys.argv[1:]:
    t0 = time.time(); C, W = SPEC[name]
    if os.path.exists(f'{OUT}/{name}.parquet'): continue
    if W == 'eval': R = build(C, EE, ER)
    else:
        T_ev = first_events(TE, C, W); R = build(C, T_ev, TR.join(T_ev.select('eid'), on='eid', how='semi'))
    R.write_parquet(f'{OUT}/{name}.parquet'); print(name, R.shape, f'{time.time()-t0:.0f}s', flush=True)
