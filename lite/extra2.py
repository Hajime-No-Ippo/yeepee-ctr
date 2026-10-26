"""Extra features #2 (leak-free; everything from ts < C):
 - ridge 'taste vector' per user on frozen SVD k64 item vectors (clicks y=1 incl. pre-C click log, skipped impressions y=0 or -0.5)
 - user state: time since last event/click, event-level click rate, recent-event click streaks, events in last 24h before C
Usage: python extra2.py NAME... -> v2work/feat_x2/NAME.parquet"""
import sys, os, time, numpy as np, polars as pl
from scipy import sparse
from sklearn.utils.extmath import randomized_svd
sys.path.insert(0, '/workspace/huawei/lite/v3')
from features import load, click_log
from split import J20, DAY, first_events
OUT = '/workspace/huawei/lite/feat_x2'; os.makedirs(OUT, exist_ok=True)
TE, TR, EE, ER, NI, NU = load('/workspace/huawei/lite/v3/data')
SPEC = {**{f'f{k}': (J20 + k * DAY, DAY) for k in range(1, 7)}, 'val25': (J20 + 5 * DAY, None), 'val26': (J20 + 6 * DAY, None),
        'val22': (J20 + 2 * DAY, None), 'eval': (J20 + 7 * DAY, 'eval'),
        **{f'e{k}': (J20 + k * DAY, (5 - k) * DAY) for k in range(1, 5)}, **{f'E{k}': (J20 + k * DAY, (7 - k) * DAY) for k in range(1, 7)}, **{f'd{k}': (J20 + k * DAY, (6 - k) * DAY) for k in range(1, 6)}}
def nrm(M): return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
def ridge_users(Wm, Ym, V, lam):
    """per-user solve (V_I^T W V_I + lam I) w = V_I^T (W*y). Wm, Ym: csr users x items (weights, weight*target)."""
    k = V.shape[1]; A = np.zeros((Wm.shape[0], k, k), np.float32)
    for a in range(k):
        cols = Wm @ (V * V[:, a:a + 1])          # (NU, k): sum_i w_ui v_ia v_ib
        A[:, a, :] = cols
    A += lam * np.eye(k, dtype=np.float32)[None]
    b = np.asarray(Ym @ V, dtype=np.float32)
    return np.linalg.solve(A, b[:, :, None])[:, :, 0]
def build(C, T_ev, T_rows):
    P_ev = TE.filter(pl.col('ts') < C)
    P_rows = TR.join(P_ev.select('eid', 'u', 'ts'), on='eid', how='inner')
    CL = click_log(C, P_ev, P_rows, T_ev)            # user-level: train data + each target row's own history
    CLg = click_log(C, P_ev, P_rows, T_ev.head(0))   # global/item-level: train data only
    assert CL['ts'].max() < C and P_rows['ts'].max() < C
    R = T_rows.join(T_ev.select('eid', 'u', 'ts'), on='eid', how='inner').select('rowid', 'eid', 'u', 'ii', 'ts')
    uu = R['u'].to_numpy().astype(np.int64); ii = R['ii'].to_numpy().astype(np.int64)
    # SVD exactly as v3 (tf-idf, row-normalised user x item click matrix)
    cnt = CLg.group_by('u', 'ii').agg(pl.len().alias('n'))
    Xc = sparse.csr_matrix((np.ones(cnt.height, np.float32), (cnt['u'].to_numpy(), cnt['ii'].to_numpy())), shape=(NU, NI))
    df_ = np.asarray(Xc.sum(0)).ravel(); idf = np.log((NU + 1) / (df_ + 1)).astype(np.float32)
    Xc = Xc @ sparse.diags(idf); Xc = sparse.diags(1 / (np.sqrt(np.asarray(Xc.multiply(Xc).sum(1)).ravel()) + 1e-6)) @ Xc
    U, S, Vt = randomized_svd(Xc, 64, random_state=0, n_iter=5); Vn = nrm(Vt.T * np.sqrt(S)).astype(np.float32)
    # (user,item) labels: clicked anywhere before C -> 1 ; shown-but-never-clicked -> 0
    pos = CL.select('u', 'ii').unique().with_columns(pl.lit(1.0).alias('y'))
    neg = P_rows.filter(pl.col('clicked') == 0).select('u', 'ii').unique().join(pos.select('u', 'ii'), on=['u', 'ii'], how='anti').with_columns(pl.lit(0.0).alias('y'))
    L = pl.concat([pos, neg])
    lu = L['u'].to_numpy(); li = L['ii'].to_numpy(); ly = L['y'].to_numpy().astype(np.float32)
    Wm = sparse.csr_matrix((np.ones(len(ly), np.float32), (lu, li)), shape=(NU, NI))
    cand = Vn[ii]; F = {}
    for lam, yneg, nm in ((1.0, 0.0, 'l1'), (10.0, 0.0, 'l10'), (3.0, -0.5, 'l3n')):
        yy = np.where(ly > 0, 1.0, yneg).astype(np.float32)
        Ym = sparse.csr_matrix((yy, (lu, li)), shape=(NU, NI))
        Wu = ridge_users(Wm, Ym, Vn, lam)
        F[f'r_{nm}'] = (Wu[uu] * cand).sum(1)
    # user state features
    ue = P_ev.select('eid', 'u', 'ts').join(P_rows.group_by('eid').agg(pl.col('clicked').sum().alias('nc')), on='eid', how='left') \
             .with_columns(pl.col('nc').fill_null(0)).sort('u', 'ts')
    us = ue.group_by('u').agg(pl.col('ts').last().alias('last_ev'), (pl.col('nc') > 0).mean().alias('s_evrate'), pl.len().alias('s_nev'),
                              pl.col('nc').tail(1).sum().alias('s_clk_last1'), pl.col('nc').tail(3).sum().alias('s_clk_last3'),
                              (pl.col('nc').tail(5) > 0).sum().alias('s_evclk_last5'), (pl.col('ts') >= C - DAY).sum().alias('s_nev24h'),
                              pl.col('ts').filter(pl.col('nc') > 0).max().alias('last_clkev'))
    lc = CL.group_by('u').agg(pl.col('ts').max().alias('last_click'), (pl.col('ts') >= C - DAY).sum().alias('s_clk24h'))
    E = T_ev.select('eid', 'u', 'ts').join(us, on='u', how='left').join(lc, on='u', how='left').with_columns(
        ((pl.col('ts') - pl.col('last_ev')) / 3600).alias('s_since_ev_h'), ((pl.col('ts') - pl.col('last_click')) / 3600).alias('s_since_clk_h'),
        ((pl.col('ts') - pl.col('last_clkev')) / 3600).alias('s_since_clkev_h')).drop('u', 'ts', 'last_ev', 'last_click', 'last_clkev')
    R = R.with_columns([pl.Series(c, v.astype(np.float32)) for c, v in F.items()])
    R = R.with_columns([((pl.col(c) - pl.col(c).mean().over('eid')) / (pl.col(c).std().over('eid') + 1e-6)).alias(c + '_z') for c in F] +
                       [(pl.col(c) - pl.col(c).max().over('eid')).alias(c + '_dmax') for c in F])
    R = R.join(E, on='eid', how='left')
    return R.drop('eid', 'u', 'ii', 'ts')
for name in sys.argv[1:]:
    t0 = time.time(); C, W = SPEC[name]
    if W == 'eval': R = build(C, EE, ER)
    else:
        T_ev = first_events(TE, C, W); R = build(C, T_ev, TR.join(T_ev.select('eid'), on='eid', how='semi'))
    R.write_parquet(f'{OUT}/{name}.parquet'); print(name, R.shape, f'{time.time()-t0:.0f}s', flush=True)
