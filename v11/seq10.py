"""v10 seq10.py = copy of shipped v8 seq.py + extra outputs for LightGBM stacking (attention stats, user/candidate dot+cos, within-event rank/gap);
fit() reseeds torch per cutoff so each fit is order-independent. Encoder variant via SQ_ENC (din|gru|rec).
v8 sequence model (shipped config: last 500k train events < C, 3 epochs, batch 256, <=6000 steps, NP 3, no slate) (DIN-style target attention), refit per cutoff C on train events ts<C only.
Tokens per event: own click history h (<=20, flag 1) + slates of the user's last 3 TRAIN events with ts < min(event ts, C)
(30 impressions, flag 2 = shown/not clicked, 3 = clicked), each = item emb + flag emb + log2 time-gap-bucket emb.
Candidate attends over tokens (2 heads, dh 16, learned null key); MLP([q, pooled, ctx]) + item bias -> logit; BCE on all 10 slate rows.
Target tables only use their own history + train events < C (no aggregates over unlabelled target/eval rows).
Usage: python seq.py NAME... [--frac 0.25]   -> feat_sq/NAME.parquet (rowid, sq_logit, sq_z, sq_dm, sq_lse)"""
import sys, os, time, argparse, numpy as np, polars as pl, torch, torch.nn as nn
D = os.environ.get('SQ_WORK', '@@WORK@@')
sys.path.insert(0, f'{D}/v3')
from features import load, click_log
from scipy import sparse
from sklearn.utils.extmath import randomized_svd
from split import J20, DAY, first_events
ap = argparse.ArgumentParser(); ap.add_argument('names', nargs='+'); ap.add_argument('--frac', type=float, default=1.0)
ap.add_argument('--ntr', type=int, default=int(os.environ.get('SQ_NTR', 500000))); ap.add_argument('--ep', type=int, default=int(os.environ.get('SQ_EP', 3)))
ap.add_argument('--d', type=int, default=32); ap.add_argument('--out', default=os.environ.get('SQ_OUT', f'{D}/feat_sq'))
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
NT = int(os.environ.get('SQ_THREADS', 4)); torch.set_num_threads(NT); torch.manual_seed(0); np.random.seed(0)
DATA = f'{D}/v3/data'
TE, TR, EE, ER, NI, NU = load(DATA)
SPEC = {**{f'f{k}': (J20 + k * DAY, DAY) for k in range(1, 7)}, 'val23': (J20 + 3 * DAY, None), 'val25': (J20 + 5 * DAY, None), 'val26': (J20 + 6 * DAY, None), 'eval': (J20 + 7 * DAY, 'eval')}
ENC = os.environ.get('SQ_ENC', 'din'); HL, NP = 20, int(os.environ.get('SQ_NP', 3)); SLATE = int(os.environ.get('SQ_SLATE', 0))
def slates(EV, RW):
    """events EV (eid,u,ts,tb,dt,h,h_ts) + rows -> arrays in EV order; candidate matrix [E,10], clicks, rowids."""
    R = RW.join(EV.select('eid', pl.int_range(pl.len()).alias('_e')), on='eid', how='inner').sort('_e', 'rowid')
    e = R['_e'].to_numpy(); assert (np.bincount(e, minlength=EV.height) == 10).all()
    cand = R['ii'].to_numpy().reshape(-1, 10).astype(np.int64)
    clk = R['clicked'].to_numpy().reshape(-1, 10).astype(np.float32) if 'clicked' in R.columns else None
    return cand, clk, R['rowid'].to_numpy().reshape(-1, 10)
# global train index sorted by (u, ts, eid): used for prior impressions
TS = TE.sort('u', 'ts', 'eid'); T_u = TS['u'].to_numpy().astype(np.int64); T_ts = TS['ts'].to_numpy()
T_cand, T_clk, _ = slates(TS, TR); KEY = T_u * (1 << 32) + T_ts
def tokens(EV, C):
    """[E, HL+NP*10] item / flag / gap-bucket tokens + ctx ints."""
    n = EV.height; u = EV['u'].to_numpy().astype(np.int64); ts = EV['ts'].to_numpy()
    it = np.full((n, HL + NP * 10), NI, np.int64); fl = np.zeros_like(it); gp = np.zeros_like(it)
    hh = EV['h'].to_list(); ht = EV['h_ts'].to_list()
    for i in range(n):
        l = hh[i]
        if l:
            l = l[-HL:]; t = np.asarray(ht[i][-HL:]); k = len(l)
            it[i, :k] = l; fl[i, :k] = 1; gp[i, :k] = 1 + np.clip(np.log2(np.maximum(ts[i] - t, 60) / 60), 0, 19).astype(np.int64)
    lim = np.minimum(ts, C); pos = np.searchsorted(KEY, u * (1 << 32) + lim, 'left') - 1   # last train event with ts < lim
    npr = np.zeros(n, np.int64)
    for j in range(NP):
        p = pos - j; ok = (p >= 0) & (T_u[np.maximum(p, 0)] == u); p = np.maximum(p, 0)
        assert (T_ts[p][ok] < lim[ok]).all() and (T_ts[p][ok] < C).all()
        s = slice(HL + 10 * j, HL + 10 * (j + 1))
        it[:, s] = np.where(ok[:, None], T_cand[p], NI); fl[:, s] = np.where(ok[:, None], 2 + T_clk[p].astype(np.int64), 0)
        g = 1 + np.clip(np.log2(np.maximum(ts - T_ts[p], 60) / 60), 0, 19).astype(np.int64); gp[:, s] = np.where(ok, g, 0)[:, None]
        npr += ok
    ctx = np.stack([EV['tb'].to_numpy().astype(np.int64), EV['dt'].to_numpy().astype(np.int64), (fl[:, :HL] > 0).sum(1), npr], 1)
    return it, fl, gp, ctx
def ppmi_E(C, k=64):
    P_ev = TE.filter(pl.col('ts') < C); P_rows = TR.join(P_ev.select('eid', 'u', 'ts'), on='eid', how='inner')
    CLg = click_log(C, P_ev, P_rows, P_ev.head(0)); assert CLg['ts'].max() < C
    s_ = CLg.select('u', pl.col('ii').cast(pl.Int64)).to_numpy(); u, it = s_[:, 0], s_[:, 1]; A = sparse.csr_matrix((NI, NI), dtype=np.float32)
    for d in range(1, 6):
        m = u[d:] == u[:-d]; x, y = it[:-d][m], it[d:][m]; ok = x != y; x, y = x[ok], y[ok]; w = np.full(len(x), 1.0 / d, np.float32)
        A = A + sparse.csr_matrix((w, (x, y)), shape=(NI, NI)) + sparse.csr_matrix((w, (y, x)), shape=(NI, NI))
    A = A.tocoo(); r = np.asarray(A.sum(1)).ravel(); c = np.asarray(A.sum(0)).ravel(); T = A.data.sum()
    pmi = np.log(A.data * T / (r[A.row] * c[A.col])) - np.log(5); k_ = pmi > 0
    Mx = sparse.csr_matrix((pmi[k_].astype(np.float32), (A.row[k_], A.col[k_])), shape=(NI, NI))
    U, S, _ = randomized_svd(Mx, k, random_state=0, n_iter=5); V = U * np.sqrt(S); V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    return np.vstack([V, np.zeros((1, k))]).astype(np.float32)
class M(nn.Module):
    def __init__(s, EMB, d=32, H=2, dh=16):
        super().__init__(); s.H, s.dh = H, dh
        s.E0 = nn.Embedding.from_pretrained(torch.from_numpy(EMB), freeze=True); s.P = nn.Linear(EMB.shape[1], d); s.S = nn.Embedding(2, H)
        s.E = lambda x: s.P(s.E0(x)); s.F = nn.Embedding(4, d); s.G = nn.Embedding(21, d);
        s.c = nn.ModuleList([nn.Embedding(4, 4), nn.Embedding(2, 2), nn.Embedding(HL + 1, 4), nn.Embedding(NP + 1, 2)])
        s.q = nn.Linear(d, H * dh); s.k = nn.Linear(d, H * dh); s.v = nn.Linear(d, H * dh); s.null = nn.Parameter(torch.zeros(1, 1, H, dh))
        if ENC == 'rec': s.R = nn.Embedding(21, d)
        if ENC == 'gru': s.gru = nn.GRU(d, d, batch_first=True)
        s.mlp = nn.Sequential(nn.Linear(d * (1 + 2 * SLATE) + H * dh * (1 + SLATE) + 12, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(s, it, fl, gp, ctx, cand, extra=False):
        B, T = it.shape; K = s.E(it) + s.F(fl) + s.G(gp); q = s.E(cand)                        # (B,T,d), (B,10,d)
        if ENC == 'rec': K = K + s.R(gp)                                                    # recency-aware keys (extra gap emb)
        if ENC == 'gru':   # GRU over tokens in reverse-recency order -> contextualised keys
            K = K + s.gru(K)[0]
        Q = s.q(q).view(B, 10, s.H, s.dh); Kk = s.k(K).view(B, T, s.H, s.dh); V = s.v(K).view(B, T, s.H, s.dh)
        same = (cand[:, :, None] == it[:, None, :]).long(); att = torch.einsum('bchd,bthd->bcht', Q, Kk) / s.dh ** 0.5 + s.S(same).permute(0, 1, 3, 2); att = att.masked_fill((fl == 0)[:, None, None, :], -1e9)
        an = (Q * s.null).sum(-1, keepdim=True) / s.dh ** 0.5; w = torch.softmax(torch.cat([att, an], -1), -1)
        P = torch.einsum('bcht,bthd->bchd', w[..., :T], V).reshape(B, 10, -1)
        cx = torch.cat([e(ctx[:, i]) for i, e in enumerate(s.c)], -1)[:, None].expand(B, 10, 12)
        f = [q, P, cx]
        if SLATE:   # within-event context: the slate's own candidates only
            qm = q.mean(1, keepdim=True).expand_as(q); Pm = P.mean(1, keepdim=True).expand_as(P); f = [q, q - qm, qm, P, P - Pm, cx]
        out = s.mlp(torch.cat(f, -1)).squeeze(-1)
        if not extra: return out
        valid = (fl > 0).float(); nv = valid.sum(1).clamp(min=1)                                # (B,T),(B,)
        wm = w.mean(2)                                                                          # head-mean weights (B,10,T+1)
        a_max = wm[..., :T].max(-1).values; a_null = wm[..., T]; a_mean = (1 - a_null) / nv[:, None]
        ent = -(wm.clamp(min=1e-9).log() * wm).sum(-1)
        vq = s.v(q)                                                                             # candidate in value space (B,10,H*dh)
        pdot = (P * vq).sum(-1); pcos = pdot / (P.norm(dim=-1) * vq.norm(dim=-1) + 1e-6)
        mu = (K * valid[..., None]).sum(1) / nv[:, None]                                        # mean-pooled user emb (B,d)
        mdot = (q * mu[:, None]).sum(-1); mcos = mdot / (q.norm(dim=-1) * mu.norm(dim=-1)[:, None] + 1e-6)
        ck = ((fl == 1) | (fl == 3)).float(); e0 = s.E0(it); eu = (e0 * ck[..., None]).sum(1); ec = s.E0(cand)
        ecos = (ec * eu[:, None]).sum(-1) / (ec.norm(dim=-1) * eu.norm(dim=-1)[:, None] + 1e-6)
        return out, torch.stack([a_max, a_null, a_mean, ent, pdot, pcos, mdot, mcos, ecos], -1)
XN = ['sq_amax', 'sq_anull', 'sq_amean', 'sq_aent', 'sq_pdot', 'sq_pcos', 'sq_mdot', 'sq_mcos', 'sq_ecos']
def usel(u): return (u * 2654435761 % 2 ** 32) / 2 ** 32 < a.frac
_M = {}
def fit(C):
    if C in _M: return _M[C]
    torch.manual_seed(0); np.random.seed(0)
    t0 = time.time(); m = T_ts < C; idx = np.nonzero(m)[0]; idx = idx[np.argsort(T_ts[idx], kind='stable')][-a.ntr:]   # train on all users; --frac only subsamples target users
    it, fl, gp, ctx = tokens(TS[idx], C); cand = T_cand[idx]; y = T_clk[idx]; tk = time.time() - t0
    X = [torch.from_numpy(z) for z in (it, fl, gp, ctx, cand)]; Y = torch.from_numpy(y)
    te = time.time(); EMB = ppmi_E(C); print(f'  ppmi C={C} {time.time() - te:.0f}s', flush=True)
    mdl = M(EMB, a.d); opt = torch.optim.Adam(mdl.parameters(), float(os.environ.get('SQ_LR', 2e-3))); lossf = nn.BCEWithLogitsLoss(); bs = int(os.environ.get('SQ_BS', 256)); n = len(idx)
    maxst = int(os.environ.get('SQ_MAXSTEPS', 6000)); st = 0   # v8: hard cap on optimizer steps -> bounded eval-time fit
    for ep in range(a.ep):
        perm = torch.randperm(n); tl = 0
        for b in range(0, n, bs):
            if st >= maxst: break
            st += 1; j = perm[b:b + bs]; opt.zero_grad(); l = lossf(mdl(*[x[j] for x in X]), Y[j]); l.backward(); opt.step(); tl += l.item() * len(j)
        print(f'  C={C} ep{ep} loss {tl / n:.4f} n={n} steps={st} {time.time() - t0:.0f}s', flush=True)
    mdl.eval(); _M[C] = mdl; print(f'  fit C={C} tok {tk:.0f}s total {time.time() - t0:.0f}s', flush=True); return mdl
def score(mdl, EV, C, cand):
    it, fl, gp, ctx = tokens(EV, C); out = []; ex = []
    with torch.no_grad():
        for b in range(0, len(cand), 8192):
            o, e = mdl(*[torch.from_numpy(z[b:b + 8192]) for z in (it, fl, gp, ctx, cand)], extra=True); out.append(o.numpy()); ex.append(e.numpy())
    return np.concatenate(out), np.concatenate(ex)
TIM = {}
for name in a.names:
    t0 = time.time(); name, _, mn = name.partition('@'); C, W = SPEC[name]; CM = SPEC[mn][0] if mn else C   # 'f4@f3': score f4 rows (history at C_f4) with the model fit at C_f3 < C_f4
    assert CM <= C
    if W == 'eval': EV, RW = EE, ER
    else: EV = first_events(TE, C, W); RW = TR.join(EV.select('eid'), on='eid', how='semi')
    if a.frac < 1: EV = EV.filter(pl.Series(usel(EV['u'].to_numpy().astype(np.int64))))
    cand, _, rid = slates(EV, RW); S, X = score(fit(CM), EV, C, cand)
    z = (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-6); lse = np.log(np.exp(S).sum(1, keepdims=True)).repeat(10, 1)
    rk = (np.argsort(np.argsort(-S, 1, kind='stable'), 1) + 1).astype(np.float32)        # 1 = event top
    gap = S - S.max(1, keepdims=True); gap2 = S - np.sort(S, 1)[:, -2:-1]                    # gap to top / to runner-up
    cols = {'rowid': rid.ravel(), 'sq_logit': S.ravel(), 'sq_z': z.ravel(), 'sq_dm': (S - S.mean(1, keepdims=True)).ravel(), 'sq_lse': lse.ravel(),
            'sq_rank': rk.ravel(), 'sq_gap': gap.ravel(), 'sq_gap2': gap2.ravel()}
    for i, n_ in enumerate(XN): cols[n_] = X[..., i].ravel()
    for n_ in ('sq_pcos', 'sq_mcos', 'sq_ecos'):   # within-event demeaned versions
        v = X[..., XN.index(n_)]; cols[n_ + '_dm'] = (v - v.mean(1, keepdims=True)).ravel()
    df = pl.DataFrame(cols).with_columns(pl.exclude('rowid').cast(pl.Float32))
    df.write_parquet(f'{a.out}/{name}.parquet'); print(name, df.shape, f'{time.time() - t0:.0f}s', flush=True)
