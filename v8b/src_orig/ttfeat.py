"""data_dig_v2 features, refit per cutoff C on train only (events ts < C); eval fitted at J27 on all train.
(A) row-scaled exact SVD: binary user x item of ALL pre-C history clicks, diag(1/sqrt(len_u)) M diag(1/sqrt(pop+5)), svds k16 rs0,
    comps 2..13 * s -> tA_cos = cos(candidate, own-history centroid) (+ _z, _dmax within event).
(B) two-tower conditional logit, tied item vectors initialised from L2-normalised (A) rows, u = mean of last<=20 own-history vectors,
    score u.v, 10-way softmax CE on clicked train events (ts<C), d12, Adam 1e-3, 6 ep, batch 2048, seed 0, no item bias, no null.
    Features: tB_dot, and over the event's own 10 candidates: tB_lse, tB_mean, tB_dmean (= dot - mean), plus tB_dot_z.
No aggregates over target/eval rows: the event features use only that event's own slate. Usage: python ttfeat.py NAME..."""
import sys, os, time, numpy as np, polars as pl, scipy.sparse as sp, torch
from scipy.sparse.linalg import svds
sys.path.insert(0, '@@WORK@@/v3')
from features import load
from split import J20, DAY, first_events
TE, TR, EE, ER, NI, NU = load('@@WORK@@/v3/data')
OUT = os.environ.get('TT_OUT', '@@WORK@@/feat_tt'); os.makedirs(OUT, exist_ok=True)
NT = int(os.environ.get('TT_THREADS', 4)); torch.set_num_threads(NT)
SPEC = {**{f'f{k}': (J20 + k * DAY, DAY) for k in range(1, 7)}, 'val25': (J20 + 5 * DAY, None), 'val26': (J20 + 6 * DAY, None)}
def nrm(V): return V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
def hmat(T_ev, L=20):
    hh = T_ev['h'].to_list(); Hm = np.full((len(hh), L), NI, np.int64)
    for i, l in enumerate(hh):
        if l: l = l[-L:]; Hm[i, :len(l)] = l
    return Hm
_M = {}
def fit(C):
    if C in _M: return _M[C]
    t0 = time.time(); P_ev = TE.filter(pl.col('ts') < C)
    h = P_ev.select('u', 'h').explode('h').drop_nulls('h').select('u', pl.col('h').cast(pl.Int64).alias('ii')).unique()
    M = sp.csr_matrix((np.ones(h.height, np.float32), (h['u'].to_numpy(), h['ii'].to_numpy())), shape=(NU, NI))
    pop = np.asarray(M.sum(0)).ravel(); ln = np.asarray(M.sum(1)).ravel()
    M = (sp.diags(1 / np.sqrt(np.maximum(ln, 1))) @ M @ sp.diags(1 / np.sqrt(pop + 5))).tocsr()
    _, s, vt = svds(M, k=16, random_state=0); o = np.argsort(-s); s, vt = s[o], vt[o]
    VA = (vt[1:13].T * s[1:13]).astype(np.float32)
    # two-tower on clicked train events before C
    R = TR.join(P_ev.select('eid'), on='eid', how='inner').sort('eid', 'rowid')
    G = R.group_by('eid', maintain_order=True).agg('ii', 'clicked').filter(pl.col('ii').list.len() == 10)
    G = G.join(P_ev.select('eid', 'h', 'ts'), on='eid').sort('ts', 'eid')
    clk = np.array(G['clicked'].to_list(), np.int8); keep = clk.sum(1) > 0
    cand = torch.from_numpy(np.array(G['ii'].to_list(), np.int64)[keep]); lab = torch.from_numpy(clk[keep].argmax(1).astype(np.int64))
    Ht = torch.from_numpy(hmat(G.filter(pl.Series(keep))))
    torch.manual_seed(0); np.random.seed(0)
    P = torch.nn.Embedding(NI + 1, 12, padding_idx=NI)
    with torch.no_grad(): P.weight[:NI] = torch.from_numpy(nrm(VA)); P.weight[NI] = 0
    opt = torch.optim.Adam(P.parameters(), lr=1e-3); N = len(lab)
    def uvec(H):
        m = (H < NI).float(); return (P(H) * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp(min=1)
    for ep in range(6):
        perm = torch.randperm(N); tot = 0.0
        for i in range(0, N, 2048):
            j = perm[i:i + 2048]; sc = (uvec(Ht[j])[:, None, :] * P(cand[j])).sum(-1)
            loss = torch.nn.functional.cross_entropy(sc, lab[j]); opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad(): P.weight[NI] = 0
            tot += loss.item() * len(j)
        print(f'  C {C} ep{ep} loss {tot / N:.4f}', flush=True)
    VB = P.weight.detach().numpy().copy()
    print(time.strftime('%T'), 'C', C, 'train events', N, f'{time.time() - t0:.0f}s', flush=True)
    _M[C] = (VA, VB); return _M[C]
def build(C, T_ev, T_rows):
    VA, VB = fit(C)
    E = T_ev.select('eid', 'h').with_row_index('k')
    R = T_rows.select('rowid', 'eid', 'ii').join(E.select('eid', 'k'), on='eid', how='left')
    rk = R['k'].to_numpy(); ri = R['ii'].to_numpy().astype(np.int64)
    # (A) all-history centroid cosine
    V = nrm(VA); H = E.select('k', 'h').explode('h').drop_nulls('h'); hc = np.zeros((E.height, V.shape[1]))
    np.add.at(hc, H['k'].to_numpy(), V[H['h'].to_numpy().astype(np.int64)]); hcr = hc[rk]; nr = np.linalg.norm(hcr, axis=1)
    a = np.where(nr > 0, (V[ri] * hcr).sum(1) / (nr + 1e-9), np.nan)
    # (B) tower dot with u = mean of last <= 20 own-history vectors
    Hm = hmat(E); m = (Hm < NI); Vp = np.vstack([VB[:NI], np.zeros((1, VB.shape[1]), VB.dtype)])
    u = (Vp[Hm] * m[..., None]).sum(1) / np.maximum(m.sum(1, keepdims=True), 1)
    dot = np.where(m.sum(1)[rk] > 0, (u[rk] * VB[ri]).sum(1), np.nan)
    R = R.select('rowid', 'eid').with_columns(pl.Series('tA_cos', a.astype(np.float32)).fill_nan(None), pl.Series('tB_dot', dot.astype(np.float32)).fill_nan(None))
    R = R.with_columns(((pl.col('tA_cos') - pl.col('tA_cos').mean().over('eid')) / (pl.col('tA_cos').std().over('eid') + 1e-6)).alias('tA_z'),
                       (pl.col('tA_cos') - pl.col('tA_cos').max().over('eid')).alias('tA_dmax'),
                       pl.col('tB_dot').exp().sum().over('eid').log().alias('tB_lse'), pl.col('tB_dot').mean().over('eid').alias('tB_mean'),
                       (pl.col('tB_dot') - pl.col('tB_dot').mean().over('eid')).alias('tB_dmean'),
                       ((pl.col('tB_dot') - pl.col('tB_dot').mean().over('eid')) / (pl.col('tB_dot').std().over('eid') + 1e-6)).alias('tB_dot_z'))
    return R.drop('eid')
for n in sys.argv[1:]:
    t0 = time.time()
    if n == 'eval': R = build(J20 + 7 * DAY, EE, ER)
    else:
        C, W = SPEC[n]; T_ev = first_events(TE, C, W); R = build(C, T_ev, TR.join(T_ev.select('eid'), on='eid', how='semi'))
    R.write_parquet(f'{OUT}/{n}.parquet'); print(n, R.shape, f'{time.time() - t0:.0f}s', flush=True)
