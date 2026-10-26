"""A1: single-feature screen of the user contrast embedding on val25 (no LightGBM).
Config: k x weighting x beta (Vn = nrm(Vt.T * S^beta)) x alpha (nrm(nC - alpha*nN)). Item side from CLg only; user side = user's own clicks (CL) + skips.
Usage: python con_sweep.py pass1|pass2   -> out/con_sweep.jsonl (skips configs already logged)"""
import sys, os, json, time, numpy as np, polars as pl
from scipy import sparse
from sklearn.utils.extmath import randomized_svd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, '/workspace/huawei/v3')
from features import load, click_log
from split import J20, DAY, first_events
D='/workspace/huawei/v5'; LOG=f'{D}/out/con_sweep.jsonl'
TE, TR, EE, ER, NI, NU = load('/workspace/huawei/v3/data')
def nrm(M): return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
def csr(u, i, w): return sparse.csr_matrix((np.asarray(w, np.float32), (u, i)), shape=(NU, NI))
def rownorm(X): return sparse.diags(1 / (np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel()) + 1e-6)) @ X
C = J20 + 5 * DAY; T_ev = first_events(TE, C, None); T_rows = TR.join(T_ev.select('eid'), on='eid', how='semi')
P_ev = TE.filter(pl.col('ts') < C); P_rows = TR.join(P_ev.select('eid', 'u', 'ts'), on='eid', how='inner')
CL = click_log(C, P_ev, P_rows, T_ev); CLg = click_log(C, P_ev, P_rows, T_ev.head(0))
assert CL['ts'].max() < C and CLg['ts'].max() < C
R = T_rows.join(T_ev.select('eid', 'u'), on='eid', how='inner').select('rowid', 'eid', 'u', 'ii', 'clicked')
uu = R['u'].to_numpy().astype(np.int64); ii = R['ii'].to_numpy().astype(np.int64); y = R['clicked'].to_numpy(); eid = R['eid'].to_numpy()
ref = R.select('rowid').join(pl.read_parquet(f'{D}/feat_x/val25.parquet', columns=['rowid', 'x_con05_k32']), on='rowid', how='left', maintain_order='left')['x_con05_k32'].to_numpy()
cg = CLg.group_by('u', 'ii').agg(pl.len().alias('n')); gu = cg['u'].to_numpy(); gi = cg['ii'].to_numpy(); gn = cg['n'].to_numpy().astype(np.float32)
cnt = CL.group_by('u', 'ii').agg(pl.len().alias('n')); Sc = csr(cnt['u'].to_numpy(), cnt['ii'].to_numpy(), np.ones(cnt.height))
neg = P_rows.filter(pl.col('clicked') == 0); Sn = csr(neg['u'].to_numpy(), neg['ii'].to_numpy(), np.ones(neg.height))
B = csr(gu, gi, np.ones(len(gu))); df_ = np.asarray(B.sum(0)).ravel(); idf = np.log((NU + 1) / (df_ + 1)).astype(np.float32)
def item_matrix(w):
    if w == 'tfidf': return rownorm(B @ sparse.diags(idf))
    if w == 'bin': return rownorm(B.copy())
    if w == 'logidf': return rownorm(csr(gu, gi, np.log1p(gn)) @ sparse.diags(idf))
    if w == 'bm25':
        k1, b = 1.2, 0.75; dl = np.bincount(gu, weights=gn, minlength=NU); avg = dl[dl > 0].mean()
        Nd = int((dl > 0).sum()); bidf = np.log((Nd - df_ + 0.5) / (df_ + 0.5) + 1).astype(np.float32)
        tf = gn * (k1 + 1) / (gn + k1 * (1 - b + b * dl[gu] / avg))
        return rownorm(csr(gu, gi, tf) @ sparse.diags(bidf))
# per-event structures for within-event AUC
o = np.argsort(eid, kind='stable'); eo = eid[o]; yo = y[o]; starts = np.r_[0, np.flatnonzero(eo[1:] != eo[:-1]) + 1, len(eo)]
def within(s):
    so = s[o]; a = []
    for i in range(len(starts) - 1):
        yy = yo[starts[i]:starts[i + 1]]
        if 0 < yy.sum() < len(yy): a.append(roc_auc_score(yy, so[starts[i]:starts[i + 1]]))
    return float(np.mean(a))
def zscore(s):
    d = pl.DataFrame({'e': eid, 's': s}).with_columns(((pl.col('s') - pl.col('s').mean().over('e')) / (pl.col('s').std().over('e') + 1e-6)).alias('z'))
    return d['z'].to_numpy()
done = set()
if os.path.exists(LOG): done = {json.loads(l)['name'] for l in open(LOG)}
def run(cfgs):
    svd = {}
    for k, w, beta, alpha in cfgs:
        name = f'x_con{str(alpha).replace(".", "")}_k{k}_{w}{str(beta).replace(".", "")}'
        if name in done: continue
        t0 = time.time()
        if (k, w) not in svd: svd = {(k, w): randomized_svd(item_matrix(w), k, random_state=0, n_iter=5)}
        U, S, Vt = svd[(k, w)]
        Vn = nrm(Vt.T * S ** beta); nC = nrm(np.asarray(Sc @ Vn)); nN = nrm(np.asarray(Sn @ Vn))
        f = (nrm(nC - alpha * nN)[uu] * Vn[ii]).sum(1); z = zscore(f)
        r = dict(name=name, k=k, w=w, beta=beta, alpha=alpha, auc_raw=round(roc_auc_score(y, f), 5), auc_z=round(roc_auc_score(y, z), 5),
                 within=round(within(f), 5), corr_ref=round(float(np.corrcoef(f, ref)[0, 1]), 4), runtime_s=round(time.time() - t0, 1))
        print(json.dumps(r), flush=True); open(LOG, 'a').write(json.dumps(r) + '\n'); done.add(name)
        np.save(f'{D}/out/cs_{name}.npy', f.astype(np.float32))
if sys.argv[1] == 'pass1':
    run([(k, w, b, 0.5) for k in (32, 16, 24, 48) for w in ('tfidf', 'bin', 'bm25', 'logidf') for b in (0.5, 0, 1)])
else:
    rows = [json.loads(l) for l in open(LOG)]; p1 = sorted([r for r in rows if r['alpha'] == 0.5], key=lambda r: -r['within'])[:6]
    run(sorted([(r['k'], r['w'], r['beta'], a) for r in p1 for a in (0.25, 1)], key=lambda c: (c[0], c[1])))
np.save(f'{D}/out/cs_rowid_val25.npy', R['rowid'].to_numpy())
