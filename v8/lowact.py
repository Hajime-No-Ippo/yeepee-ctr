"""S2 (v8): low-activity / multi-window user features. Train-only stats (ts < C) + each target row's own history (via click_log).
uw_*: user events / impression rows / clicks in the last 1h/1d/7d before the target event ts (only data with ts < C is visible).
la_*: item x user-activity-bin CTR (m=20 smoothed to item CTR, item CTR smoothed to global), activity-bin x item-popularity-bin CTR.
Usage: python lowact.py NAME... -> v8/feat_la/NAME.parquet"""
import sys, os, time, numpy as np, polars as pl
sys.path.insert(0, '/workspace/huawei/v3')
from features import load, click_log
from split import J20, DAY, first_events
OUT = os.environ.get('LA_OUT', '/workspace/huawei/v8/feat_la'); os.makedirs(OUT, exist_ok=True)
TE, TR, EE, ER, NI, NU = load('/workspace/huawei/v3/data')
SPEC = {**{f'f{k}': (J20 + k * DAY, DAY) for k in range(1, 7)}, 'val23': (J20 + 3 * DAY, None), 'val25': (J20 + 5 * DAY, None), 'val26': (J20 + 6 * DAY, None)}
W = {'1h': 3600, '1d': DAY, '7d': 7 * DAY}; M = 20.0
EDGES = np.array([1, 3, 6, 11, 21])          # activity bins: 0,1-2,3-5,6-10,11-20,21+
def cnt_in(keys_u, keys_t, wts, qu, qt, w):
    """sum of wts for (u, t) with qt-w <= t < qt (keys sorted by u,t)."""
    K = keys_u.astype(np.int64) * (1 << 32) + keys_t.astype(np.int64); cs = np.concatenate([[0], np.cumsum(wts)])
    hi = np.searchsorted(K, qu.astype(np.int64) * (1 << 32) + qt, 'left'); lo = np.searchsorted(K, qu.astype(np.int64) * (1 << 32) + (qt - w), 'left')
    return (cs[hi] - cs[lo]).astype(np.float32)
def build(C, T_ev, T_rows):
    P_ev = TE.filter(pl.col('ts') < C); P_rows = TR.join(P_ev.select('eid', 'u', 'ts'), on='eid', how='inner')
    CL = click_log(C, P_ev, P_rows, T_ev); assert CL['ts'].max() < C and P_rows['ts'].max() < C
    R = T_rows.join(T_ev.select('eid', 'u', 'ts'), on='eid', how='inner').select('rowid', 'eid', 'u', 'ts', 'ii')
    E = T_ev.select('eid', 'u', 'ts'); qu = E['u'].to_numpy(); qt = E['ts'].to_numpy().astype(np.int64)
    pe = P_rows.group_by('eid', 'u', 'ts').agg(pl.len().alias('n')).sort('u', 'ts')
    cl = CL.select('u', 'ts').sort('u', 'ts')
    F = {}
    for k, w in W.items():
        F[f'uw_ev_{k}'] = cnt_in(pe['u'].to_numpy(), pe['ts'].to_numpy(), np.ones(pe.height), qu, qt, w)
        F[f'uw_imp_{k}'] = cnt_in(pe['u'].to_numpy(), pe['ts'].to_numpy(), pe['n'].to_numpy().astype(np.float64), qu, qt, w)
        F[f'uw_clk_{k}'] = cnt_in(cl['u'].to_numpy(), cl['ts'].to_numpy(), np.ones(cl.height), qu, qt, w)
        F[f'uw_ctr_{k}'] = (F[f'uw_clk_{k}'] + 0.5) / (F[f'uw_imp_{k}'] + 5.0)
    F['uw_ev_r1d7d'] = (F['uw_ev_1d'] + 0.5) / (F['uw_ev_7d'] + 1.0); F['uw_clk_r1d7d'] = (F['uw_clk_1d'] + 0.5) / (F['uw_clk_7d'] + 1.0)
    E = E.with_columns([pl.Series(c, v) for c, v in F.items()]).select(['eid'] + list(F))
    # activity bins (pre-C events per user) and item popularity bins
    act = P_ev.group_by('u').agg(pl.len().alias('na')); ub = np.zeros(NU, np.int8); ub[act['u'].to_numpy()] = np.searchsorted(EDGES, act['na'].to_numpy(), 'right')
    g = P_rows['clicked'].mean()
    it = P_rows.group_by('ii').agg(pl.len().alias('n'), pl.col('clicked').sum().alias('c'))
    ictr = np.full(NI, g, np.float64); imp = np.zeros(NI); ictr[it['ii'].to_numpy()] = (it['c'].to_numpy() + M * g) / (it['n'].to_numpy() + M); imp[it['ii'].to_numpy()] = it['n'].to_numpy()
    qs = np.quantile(imp[imp > 0], np.linspace(0.1, 0.9, 9)); ib = np.where(imp > 0, 1 + np.searchsorted(qs, imp, 'right'), 0).astype(np.int8)
    Pb = P_rows.select(pl.col('ii').cast(pl.Int64), 'clicked', pl.Series('ab', ub[P_rows['u'].to_numpy()]))
    t1 = Pb.group_by('ii', 'ab').agg(pl.len().alias('n'), pl.col('clicked').sum().alias('c'))
    T1 = np.full((NI, 6), np.nan); prior = ictr[t1['ii'].to_numpy()]
    T1[t1['ii'].to_numpy(), t1['ab'].to_numpy()] = (t1['c'].to_numpy() + M * prior) / (t1['n'].to_numpy() + M)
    T1 = np.where(np.isnan(T1), ictr[:, None], T1)
    Pb = Pb.with_columns(pl.Series('pb', ib[Pb['ii'].to_numpy()]))
    t2 = Pb.group_by('ab', 'pb').agg(pl.len().alias('n'), pl.col('clicked').sum().alias('c'))
    T2 = np.full((6, 11), g); T2[t2['ab'].to_numpy(), t2['pb'].to_numpy()] = (t2['c'].to_numpy() + M * g) / (t2['n'].to_numpy() + M)
    uu = R['u'].to_numpy(); ii = R['ii'].to_numpy().astype(np.int64); a = ub[uu]
    G = {'la_actbin': a.astype(np.float32), 'la_i_ab_ctr': T1[ii, a].astype(np.float32), 'la_i_ab_lift': (T1[ii, a] / ictr[ii]).astype(np.float32),
         'la_ab_pb_ctr': T2[a, ib[ii]].astype(np.float32)}
    R = R.with_columns([pl.Series(c, v) for c, v in G.items()]).join(E, on='eid', how='left')
    R = R.with_columns([((pl.col(c) - pl.col(c).mean().over('eid')) / (pl.col(c).std().over('eid') + 1e-6)).alias(c + '_z') for c in ('la_i_ab_ctr', 'la_i_ab_lift')])
    return R.drop('eid', 'u', 'ts', 'ii')
for n in sys.argv[1:]:
    t0 = time.time()
    if n == 'eval': R = build(J20 + 7 * DAY, EE, ER)
    else:
        C, Wd = SPEC[n]; T_ev = first_events(TE, C, Wd); R = build(C, T_ev, TR.join(T_ev.select('eid'), on='eid', how='semi'))
    R.write_parquet(f'{OUT}/{n}.parquet'); print(n, R.shape, f'{time.time()-t0:.0f}s', flush=True)
