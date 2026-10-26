"""Extra leak-free features on v3 data (all stats from data with ts < C).
Usage: python extra.py NAME...   (names: f1..f6, val25, val26, eval)  -> v5/feat_x/NAME.parquet
 - con variants: k=32/128, recency-weighted (tau=2d) click/non-click centroids, item-side con (users who clicked i vs skipped i)
 - user-cluster item CTR (kmeans on user SVD vectors, 64 & 256 clusters), smoothed toward item CTR
 - within-event z-score and rank for each"""
import sys, os, time, numpy as np, polars as pl
from scipy import sparse
from sklearn.utils.extmath import randomized_svd
from sklearn.cluster import MiniBatchKMeans
sys.path.insert(0, '/workspace/huawei/v3')
from features import load, click_log
from split import J20, DAY, first_events
# v5 env flags: FIX=1 -> A0 leak fix (item side / KMeans fit from train-only Ug); NEW=1 -> Step A features (A1 sweep picks in SEL json, A2, A3), drops dup x_con05_k64
FIX = os.environ.get('FIX') == '1' or sys.argv[1:2] == ['a0check']; NEW = os.environ.get('NEW') == '1'
SEL = [tuple(c) for c in __import__('json').load(open(os.environ['SEL']))] if os.environ.get('SEL') else []   # [(k, w, beta, alpha, name), ...]
OUT = os.environ.get('OUT', '/workspace/huawei/v5/feat_x'); os.makedirs(OUT, exist_ok=True)
TE, TR, EE, ER, NI, NU = load('/workspace/huawei/v3/data')
SPEC = {**{f'f{k}': (J20 + k * DAY, DAY) for k in range(1, 7)}, 'val23': (J20 + 3 * DAY, None), 'val25': (J20 + 5 * DAY, None), 'val26': (J20 + 6 * DAY, None), 'eval': (J20 + 7 * DAY, 'eval'),
        **{f'e{k}': (J20 + k * DAY, (5 - k) * DAY) for k in range(1, 5)}, **{f'E{k}': (J20 + k * DAY, (7 - k) * DAY) for k in range(1, 7)}, **{f'd{k}': (J20 + k * DAY, (6 - k) * DAY) for k in range(1, 6)}}
def nrm(M): return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
def csr(u, i, w): return sparse.csr_matrix((w.astype(np.float32), (u, i)), shape=(NU, NI))
DBG = {}
def rownorm(X): return sparse.diags(1 / (np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel()) + 1e-6)) @ X
def sweep_feats(sel, CLg, Sc, Sn, uu, ii):
    cg = CLg.group_by('u', 'ii').agg(pl.len().alias('n')); gu = cg['u'].to_numpy(); gi = cg['ii'].to_numpy(); gn = cg['n'].to_numpy().astype(np.float32)
    B = csr(gu, gi, np.ones(len(gu))); df_ = np.asarray(B.sum(0)).ravel(); idf = np.log((NU + 1) / (df_ + 1)).astype(np.float32)
    def item_matrix(w):
        if w == 'tfidf': return rownorm(B @ sparse.diags(idf))
        if w == 'bin': return rownorm(B.copy())
        if w == 'logidf': return rownorm(csr(gu, gi, np.log1p(gn)) @ sparse.diags(idf))
        if w == 'bm25':
            k1, b = 1.2, 0.75; dl = np.bincount(gu, weights=gn, minlength=NU); avg = dl[dl > 0].mean()
            Nd = int((dl > 0).sum()); bidf = np.log((Nd - df_ + 0.5) / (df_ + 0.5) + 1).astype(np.float32)
            return rownorm(csr(gu, gi, gn * (k1 + 1) / (gn + k1 * (1 - b + b * dl[gu] / avg))) @ sparse.diags(bidf))
    out = {}
    for k, w, beta, alpha, name in sel:
        U, S, Vt = randomized_svd(item_matrix(w), k, random_state=0, n_iter=5)
        Vn = nrm(Vt.T * S ** beta); nC = nrm(np.asarray(Sc @ Vn)); nN = nrm(np.asarray(Sn @ Vn)); dv = nC - alpha * nN
        out[name] = (nrm(dv)[uu] * Vn[ii]).sum(1)
        if name == sel[0][4]: out[name + '_norm'] = np.linalg.norm(dv, axis=1)[uu]; out[name + '_dot'] = (dv[uu] * Vn[ii]).sum(1)
    return out
def build(C, T_ev, T_rows):
    P_ev = TE.filter(pl.col('ts') < C)
    P_rows = TR.join(P_ev.select('eid', 'u', 'ts'), on='eid', how='inner')
    CL = click_log(C, P_ev, P_rows, T_ev)            # user-level: train data + each target row's own history
    CLg = click_log(C, P_ev, P_rows, T_ev.head(0))   # global/item-level: train data only
    assert CL['ts'].max() < C and P_rows['ts'].max() < C
    R = T_rows.join(T_ev.select('eid', 'u'), on='eid', how='inner').select('rowid', 'eid', 'u', 'ii')
    uu = R['u'].to_numpy().astype(np.int64); ii = R['ii'].to_numpy().astype(np.int64)
    cg = CLg.group_by('u', 'ii').agg(pl.len().alias('n'))
    Xc = csr(cg['u'].to_numpy(), cg['ii'].to_numpy(), np.ones(cg.height)); Scg = Xc.copy()
    cnt = CL.group_by('u', 'ii').agg(pl.len().alias('n'), pl.col('ts').max().alias('lt'))
    cu = cnt['u'].to_numpy(); ci = cnt['ii'].to_numpy()
    df_ = np.asarray(Xc.sum(0)).ravel(); idf = np.log((NU + 1) / (df_ + 1)).astype(np.float32)
    Xc = Xc @ sparse.diags(idf); Xc = sparse.diags(1 / (np.sqrt(np.asarray(Xc.multiply(Xc).sum(1)).ravel()) + 1e-6)) @ Xc
    Xu = csr(cu, ci, np.ones(cnt.height)) @ sparse.diags(idf); Xu = sparse.diags(1 / (np.sqrt(np.asarray(Xu.multiply(Xu).sum(1)).ravel()) + 1e-6)) @ Xu
    neg = P_rows.filter(pl.col('clicked') == 0)
    nu_ = neg['u'].to_numpy(); ni_ = neg['ii'].to_numpy()
    Sc = csr(cu, ci, np.ones(cnt.height)); Sn = csr(nu_, ni_, np.ones(neg.height))
    rc = np.exp(-(C - cnt['lt'].to_numpy()) / (2 * DAY)); rn = np.exp(-(C - neg['ts'].to_numpy()) / (2 * DAY))
    Scr = csr(cu, ci, rc); Snr = csr(nu_, ni_, rn)
    F = {}
    for k in (32, 64, 128):
        U, S, Vt = randomized_svd(Xc, k, random_state=0, n_iter=5)
        Vn = nrm(Vt.T * np.sqrt(S)); Un = nrm(np.asarray(Xu @ (Vt.T / np.sqrt(S)))); cand = Vn[ii]  # user fold-in
        Ug = nrm(np.asarray(Xc @ (Vt.T / np.sqrt(S)))) if FIX else Un   # train-only (CLg) user vectors for item side / KMeans fit
        if FIX: DBG[k] = Ug
        nC = nrm(np.asarray(Sc @ Vn)); nN = nrm(np.asarray(Sn @ Vn))
        if not (NEW and k == 64): F[f'x_con05_k{k}'] = (nrm(nC - 0.5 * nN)[uu] * cand).sum(1)   # k64 = exact dup of sv_k64_con05 (A4)
        if NEW and k == 32:
            dv = (nC - 0.5 * nN); F['x_con05_k32_norm'] = np.linalg.norm(dv, axis=1)[uu]; F['x_con05_k32_dot'] = (dv[uu] * cand).sum(1)
            F['x_con05_k32_dmax'] = (nrm(dv)[uu] * cand).sum(1)   # converted to raw - event max below
            iCk = nrm(np.asarray(Scg.T @ Ug)); iNk = nrm(np.asarray(Sn.T @ Ug))
            for a_, nm in ((0.5, '05'), (1.0, '1')): F[f'x_icon{nm}_k32'] = (nrm(iCk - a_ * iNk)[ii] * Un[uu]).sum(1)
            F['x_icos_k32'] = (iCk[ii] * Un[uu]).sum(1)
        if k == 64:
            nCr = nrm(np.asarray(Scr @ Vn)); nNr = nrm(np.asarray(Snr @ Vn))
            for a, nm in ((0.5, '05'), (1.0, '1')):
                F[f'x_conr{nm}_k64'] = (nrm(nCr - a * nNr)[uu] * cand).sum(1)
            F['x_crcos_k64'] = (nCr[uu] * cand).sum(1)
            # item side: centroid of users who clicked item minus users who skipped it, vs target user vector
            iC = nrm(np.asarray(Scg.T @ Ug)); iN = nrm(np.asarray(Sn.T @ Ug))  # item side: train-only clicks (+ train-only user vectors if FIX)
            if FIX: DBG['iC'] = iC; DBG['iN'] = iN
            if NEW: F['x_icon025_k64'] = (nrm(iC - 0.25 * iN)[ii] * Un[uu]).sum(1)
            F['x_icon05_k64'] = (nrm(iC - 0.5 * iN)[ii] * Un[uu]).sum(1)
            F['x_icos_k64'] = (iC[ii] * Un[uu]).sum(1)
            # user-cluster CTR
            act = np.unique(np.concatenate([P_rows['u'].to_numpy(), cg['u'].to_numpy()]))
            for nc in (64, 256):
                km = MiniBatchKMeans(nc, random_state=0, batch_size=4096, n_init=3).fit(Ug[act])
                if FIX: DBG[f'km{nc}'] = km.cluster_centers_
                lab = km.predict(Un).astype(np.int32)
                g = float(P_rows['clicked'].mean())
                pr = P_rows.select('ii', 'clicked', pl.Series('cl', lab[P_rows['u'].to_numpy()]))
                ict = pr.group_by('ii').agg(((pl.col('clicked').sum() + 20 * g) / (pl.len() + 20)).alias('ictr'))
                cct = pr.group_by('cl', 'ii').agg(pl.col('clicked').sum().alias('c'), pl.len().alias('n'))
                q = pl.DataFrame({'cl': lab[uu], 'ii': ii.astype(np.int32), 'r': np.arange(len(uu))}).join(cct, on=['cl', 'ii'], how='left') \
                      .join(ict, on='ii', how='left').sort('r').with_columns(pl.col('ictr').fill_null(g), pl.col('c').fill_null(0), pl.col('n').fill_null(0))
                F[f'x_clctr{nc}'] = ((q['c'] + 10 * q['ictr']) / (q['n'] + 10)).to_numpy()
                F[f'x_clctr{nc}_lift'] = F[f'x_clctr{nc}'] / q['ictr'].to_numpy()
                F[f'x_climp{nc}'] = q['n'].to_numpy().astype(np.float32)
    if NEW:   # A1 picks: user contrast with alternative weighting / beta / alpha (item factors from CLg only)
        F.update(sweep_feats(SEL, CLg, Sc, Sn, uu, ii))
        F['x_nclk_cl'] = np.asarray(Sc.sum(1)).ravel()[uu]; F['x_nskip_p'] = np.asarray(Sn.sum(1)).ravel()[uu]
    R = R.with_columns([pl.Series(c, v.astype(np.float32)) for c, v in F.items()])
    if NEW: R = R.with_columns([(pl.col(c) - pl.col(c).max().over('eid')).alias(c) for c in F if c.endswith('_dmax')])
    zc = [c for c in F if not c.startswith('x_climp') and not c.endswith(('_norm', '_dot', '_dmax')) and c not in ('x_nclk_cl', 'x_nskip_p')]
    R = R.with_columns([((pl.col(c) - pl.col(c).mean().over('eid')) / (pl.col(c).std().over('eid') + 1e-6)).alias(c + '_z') for c in zc] +
                       [pl.col(c).rank('average').over('eid').alias(c + '_rk') for c in zc])
    return R.drop('eid', 'u', 'ii')
if __name__ == '__main__' and sys.argv[1:2] == ['a0check']:   # A0 acceptance: eval item side / KMeans identical with and without eval histories
    C = J20 + 7 * DAY; build(C, EE, ER); A = {k: (v.copy() if hasattr(v, 'copy') else v) for k, v in DBG.items()}
    DBG.clear(); build(C, EE.head(0), ER.head(0)); out = {}
    for k in A: out[str(k)] = float(np.abs(np.asarray(A[k]) - np.asarray(DBG[k])).max())
    assert out and set(out) >= {'iC', 'iN', 'km64', 'km256'}, out
    print('A0CHECK', __import__('json').dumps(out), flush=True); sys.exit(0)
for name in sys.argv[1:]:
    t0 = time.time(); C, W = SPEC[name]
    if W == 'eval': R = build(C, EE, ER)
    else:
        T_ev = first_events(TE, C, W); R = build(C, T_ev, TR.join(T_ev.select('eid'), on='eid', how='semi'))
    R.write_parquet(f'{OUT}/{name}.parquet'); print(name, R.shape, f'{time.time()-t0:.0f}s', flush=True)
