"""v8 (ported from v5 ppmi.py): sequence-window PPMI-SVD item embeddings from CLg only (train clicks + train-event histories, ts<C).
Usage: python ppmi.py NAME... -> WORK/feat_pp/NAME.parquet"""
import sys, os, time, json, numpy as np, polars as pl
from scipy import sparse
from sklearn.utils.extmath import randomized_svd
sys.path.insert(0, '/workspace/huawei/v8b/work/v3')
from features import load, click_log
from split import J20, DAY, first_events
OUT = '/workspace/huawei/v8b/work/feat_pp'; os.makedirs(OUT, exist_ok=True)
TE, TR, EE, ER, NI, NU = load('/workspace/huawei/v8b/work/v3/data')
SPEC = {**{f'f{k}': (J20 + k * DAY, DAY) for k in range(1, 7)}, 'val23': (J20 + 3 * DAY, None), 'val25': (J20 + 5 * DAY, None), 'val26': (J20 + 6 * DAY, None), 'eval': (J20 + 7 * DAY, 'eval')}
def nrm(M): return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
def csr(u, i, w, shape): return sparse.csr_matrix((np.asarray(w, np.float32), (u, i)), shape=shape)
def ppmi_emb(CLg, ks=(32, 64)):
    s = CLg.select('u', pl.col('ii').cast(pl.Int64)).to_numpy()   # sorted by u, ts
    u, it = s[:, 0], s[:, 1]; A = sparse.csr_matrix((NI, NI), dtype=np.float32)
    for d in range(1, 6):
        m = u[d:] == u[:-d]; a, b = it[:-d][m], it[d:][m]; ok = a != b; a, b = a[ok], b[ok]
        A = A + csr(a, b, np.full(len(a), 1.0 / d), (NI, NI)) + csr(b, a, np.full(len(a), 1.0 / d), (NI, NI))
    A = A.tocoo(); r = np.asarray(A.sum(1)).ravel(); c = np.asarray(A.sum(0)).ravel(); T = A.data.sum()
    pmi = np.log(A.data * T / (r[A.row] * c[A.col])) - np.log(5); k_ = pmi > 0
    M = sparse.csr_matrix((pmi[k_].astype(np.float32), (A.row[k_], A.col[k_])), shape=(NI, NI))
    E = {}
    for k in ks:
        U, S, Vt = randomized_svd(M, k, random_state=0, n_iter=5); E[k] = nrm(U * np.sqrt(S))
    return E, int(len(a))
def build(C, T_ev, T_rows):
    P_ev = TE.filter(pl.col('ts') < C); P_rows = TR.join(P_ev.select('eid', 'u', 'ts'), on='eid', how='inner')
    CL = click_log(C, P_ev, P_rows, T_ev); CLg = click_log(C, P_ev, P_rows, T_ev.head(0))
    assert CLg['ts'].max() < C and CL['ts'].max() < C
    # acceptance: window counts from CLg only -> target events' histories must not change CLg
    assert CLg.height == click_log(C, P_ev, P_rows, T_ev.head(0)).height
    E, _ = ppmi_emb(CLg)
    R = T_rows.join(T_ev.select('eid', 'u'), on='eid', how='inner').select('rowid', 'eid', 'u', 'ii')
    uu = R['u'].to_numpy().astype(np.int64); ii = R['ii'].to_numpy().astype(np.int64)
    cnt = CL.group_by('u', 'ii').agg(pl.len()); Sc = csr(cnt['u'].to_numpy(), cnt['ii'].to_numpy(), np.ones(cnt.height), (NU, NI))
    neg = P_rows.filter(pl.col('clicked') == 0); Sn = csr(neg['u'].to_numpy(), neg['ii'].to_numpy(), np.ones(neg.height), (NU, NI))
    F = {}
    E32, E64 = E[32], E[64]
    nC = nrm(np.asarray(Sc @ E32)); nN = nrm(np.asarray(Sn @ E32)); F['x_pp_con05_k32'] = (nrm(nC - 0.5 * nN)[uu] * E32[ii]).sum(1)
    hC = nrm(np.asarray(Sc @ E64)); F['x_pp_hcos'] = (hC[uu] * E64[ii]).sum(1)
    l3 = CL.sort('u', 'ts').group_by('u', maintain_order=True).tail(3).select('u', pl.col('ii').cast(pl.Int64))
    L = np.full((NU, 3), -1, np.int64); pos = l3.with_columns(pl.int_range(pl.len()).over('u').alias('p'))
    L[pos['u'].to_numpy(), pos['p'].to_numpy()] = pos['ii'].to_numpy()
    best = np.full(len(uu), -np.inf, np.float32); Lu = L[uu]
    for j in range(3):
        v = Lu[:, j]; ok = v >= 0; sc = np.full(len(uu), -np.inf, np.float32); sc[ok] = (E64[v[ok]] * E64[ii[ok]]).sum(1)
        best = np.maximum(best, sc)
    F['x_pp_last3'] = np.where(np.isfinite(best), best, 0.0)
    R = R.with_columns([pl.Series(c, v.astype(np.float32)) for c, v in F.items()])
    R = R.with_columns([((pl.col(c) - pl.col(c).mean().over('eid')) / (pl.col(c).std().over('eid') + 1e-6)).alias(c + '_z') for c in F] +
                       [pl.col(c).rank('average').over('eid').alias(c + '_rk') for c in F])
    return R
def table(name):
    C, W = SPEC[name]
    if W == 'eval': return build(C, EE, ER)
    T_ev = first_events(TE, C, W); return build(C, T_ev, TR.join(T_ev.select('eid'), on='eid', how='semi'))
for name in sys.argv[1:]:
    t0 = time.time(); R = table(name).drop('eid', 'u', 'ii'); R.write_parquet(f'{OUT}/{name}.parquet'); print(name, R.shape, f'{time.time()-t0:.0f}s', flush=True)
