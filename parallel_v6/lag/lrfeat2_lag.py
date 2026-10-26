"""F1 (data_dig_v1): history-centroid cosine in an exact low-rank SVD space. Train-only, rebuilt at each fold cutoff C.
Matrix: users x items, binary, rows = each user's last N distinct history clicks (from train events with ts < C),
columns scaled by 1/sqrt(pop+5); exact svds (ARPACK) k=17; item vectors = comps 2..K+1 (popularity comp dropped) * s.
Feature: cos(candidate vector, sum of own-history normalised vectors). Within-event _z and _dmax. Eval: fitted at J27 on train only.
Usage: python lrfeat.py NAME... -> feat_lr/NAME.parquet (rowid, lr{K}_cos/_z/_dmax for K in 10,12,16)"""
import sys, os, time, numpy as np, polars as pl, scipy.sparse as sp
from scipy.sparse.linalg import svds
sys.path.insert(0, '/workspace/huawei/v3')
from features import load
from split import J20, DAY, first_events
TE, TR, EE, ER, NI, NU = load('/workspace/huawei/v3/data')
N = int(os.environ.get('LR_N', 40)); OUT = '/workspace/huawei/parallel_v6/lag/feat_lr2'; os.makedirs(OUT, exist_ok=True)
KS = tuple(int(x) for x in os.environ.get('LR_KS', '10,12,16').split(','))
CON = int(os.environ.get('LR_CON', 0))
SPEC = {**{f'f{k}': (J20 + k * DAY, DAY) for k in range(1, 7)}, 'val25': (J20 + 5 * DAY, None), 'val26': (J20 + 6 * DAY, None)}
_V = {}
def vecs(C):
    if C in _V: return _V[C]
    h = (TE.filter(pl.col('ts') < C).select('u', 'h', 'h_ts').explode(['h', 'h_ts']).drop_nulls('h')
           .select('u', pl.col('h').cast(pl.Int32).alias('ii'), pl.col('h_ts').cast(pl.Int64).alias('t')).unique().sort('u', 't', 'ii'))
    h = h.with_columns(pl.int_range(pl.len()).over('u').alias('p'), pl.len().over('u').alias('L')).filter(pl.col('L') - pl.col('p') <= N).unique(['u', 'ii'])
    M = sp.csr_matrix((np.ones(h.height, np.float32), (h['u'].to_numpy(), h['ii'].to_numpy())), shape=(NU, NI))
    pop = np.asarray(M.sum(0)).ravel(); M = M.multiply(1 / np.sqrt(pop + 5)).tocsr()
    _, s, vt = svds(M, k=max(max(KS) + 1, 17), random_state=0); o = np.argsort(-s); s, vt = s[o], vt[o]
    print(time.strftime('%T'), 'C', C, 'pairs', h.height, 'sv', np.round(s[:15], 2), flush=True)
    _V[C] = (s, vt); return _V[C]
def build(C, T_ev, T_rows):
    s, vt = vecs(C)
    E = T_ev.select('eid', 'h').with_row_index('k'); kk = E.select('eid', 'k')
    H = E.select('k', 'h').explode('h').drop_nulls('h'); hk = H['k'].to_numpy(); hi = H['h'].to_numpy().astype(np.int64)
    R = T_rows.select('rowid', 'eid', 'ii').join(kk, on='eid', how='left'); rk = R['k'].to_numpy(); ri = R['ii'].to_numpy().astype(np.int64)
    F = {}
    if CON:
        P_ev = TE.filter(pl.col('ts') < C).select('eid', 'u')
        ng = TR.filter(pl.col('clicked') == 0).join(P_ev, on='eid', how='inner').select('u', 'ii').unique()
        ue = T_ev.select('u').with_row_index('k').join(ng, on='u', how='inner'); nk = ue['k'].to_numpy(); ni = ue['ii'].to_numpy().astype(np.int64)
    for K in KS:
        V = (vt[1:K + 1].T * s[1:K + 1]); V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
        hc = np.zeros((E.height, K)); np.add.at(hc, hk, V[hi])
        hcr = hc[rk]; nr = np.linalg.norm(hcr, axis=1)
        x = np.where(nr > 0, (V[ri] * hcr).sum(1) / (nr + 1e-9), np.nan)
        F[f'lr{K}_cos'] = x.astype(np.float32)
        if CON:
            nc = np.zeros((E.height, K)); np.add.at(nc, nk, V[ni])
            hn = hc / (np.linalg.norm(hc, axis=1, keepdims=True) + 1e-9); nn = nc / (np.linalg.norm(nc, axis=1, keepdims=True) + 1e-9)
            for al, nm in ((0.5, '05'), (1.0, '1')):
                d = (hn - al * nn)[rk]; dn = np.linalg.norm(d, axis=1)
                F[f'lr{K}_con{nm}'] = np.where(dn > 0, (V[ri] * d).sum(1) / (dn + 1e-9), np.nan).astype(np.float32)
    R = R.select('rowid', 'eid').with_columns([pl.Series(c, v).fill_nan(None) for c, v in F.items()])
    R = R.with_columns([((pl.col(c) - pl.col(c).mean().over('eid')) / (pl.col(c).std().over('eid') + 1e-6)).alias(c.replace('_cos', '_z') if c.endswith('_cos') else c + '_z') for c in F] +
                       [(pl.col(c) - pl.col(c).max().over('eid')).alias(c.replace('_cos', '_dmax') if c.endswith('_cos') else c + '_dmax') for c in F])
    return R.drop('eid')
for n in sys.argv[1:]:
    t0 = time.time()
    if n == 'val25lag': T_ev = first_events(TE, J20 + 5 * DAY, None); R = build(J20 + 4 * DAY, T_ev, TR.join(T_ev.select('eid'), on='eid', how='semi'))
    elif n == 'eval': R = build(J20 + 7 * DAY, EE, ER)
    else:
        C, W = SPEC[n]; T_ev = first_events(TE, C, W); R = build(C, T_ev, TR.join(T_ev.select('eid'), on='eid', how='semi'))
    R.write_parquet(f'{OUT}/{n}.parquet'); print(n, R.shape, f'{time.time()-t0:.0f}s', flush=True)
