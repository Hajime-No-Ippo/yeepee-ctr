"""Phase 2: DIN-style sequence model (PyTorch CPU). Train data only (events ts < C); each target row uses only its own history + its event's 10 candidates.
python seq.py CUT [--epochs 2] [--threads 4]  -> ens/seq_c{CUT}.parquet (rowid, seq_logit). CUT=27 -> eval."""
import sys, time, argparse, numpy as np, polars as pl, torch, torch.nn as nn
sys.path.insert(0, '/workspace/huawei/v3')
from features import load
from split import J20, DAY, first_events
from sklearn.metrics import roc_auc_score
ap = argparse.ArgumentParser(); ap.add_argument('cut', type=int); ap.add_argument('--epochs', type=int, default=2); ap.add_argument('--threads', type=int, default=4)
ap.add_argument('--dim', type=int, default=32); ap.add_argument('--bs', type=int, default=2048); ap.add_argument('--minf', type=int, default=3)
a = ap.parse_args(); torch.set_num_threads(a.threads); torch.manual_seed(0); np.random.seed(0); t0 = time.time()
def log(*x): print(f'[{(time.time()-t0)/60:.1f}m]', *x, flush=True)
TE, TR, EE, ER, NI, NU = load('/workspace/huawei/v3/data')
C = J20 + (a.cut - 20) * DAY
P_ev = TE.filter(pl.col('ts') < C); P_rows = TR.join(P_ev.select('eid'), on='eid', how='semi')
if a.cut == 27: T_ev, T_rows = EE, ER.select('rowid', 'eid', 'ii')
else:
    T_ev = first_events(TE, C, None); T_rows = TR.join(T_ev.select('eid'), on='eid', how='semi')
# vocab from train data (< C) only
cnt = np.bincount(np.concatenate([P_rows['ii'].to_numpy(), P_ev['h'].explode().drop_nulls().to_numpy()]), minlength=NI)
vid = np.ones(NI, np.int64); keep = np.where(cnt >= a.minf)[0]; vid[keep] = np.arange(2, len(keep) + 2); V = len(keep) + 2
log('vocab', V, 'train rows', P_rows.height, 'target rows', T_rows.height)
L = 20
def ev_arrays(E):
    E = E.sort('eid')
    n = E.height
    T = E.select(pl.col('h').list.tail(L).alias('h'), pl.col('h_ts').list.tail(L).alias('t'))
    lens = T['h'].list.len().to_numpy().astype(np.int64)
    NE = T.filter(pl.col('h').list.len() > 0)
    hv = NE['h'].explode().to_numpy().astype(np.int64); tv = NE['t'].explode().to_numpy().astype(np.int64)
    assert len(hv) == lens.sum() == len(tv), (len(hv), lens.sum(), len(tv))
    idx_ev = np.repeat(np.arange(n), lens); starts = np.cumsum(lens) - lens
    pos = np.arange(len(idx_ev)) - starts[idx_ev] + (L - lens)[idx_ev]
    H = np.zeros((n, L), np.int64); G = np.zeros((n, L), np.int64)
    ts = E['ts'].to_numpy()
    gap = np.clip(np.floor(np.log2(1 + np.maximum(ts[idx_ev] - tv, 0) / 60)), 0, 20).astype(np.int64) + 1
    H[idx_ev, pos] = vid[hv]; G[idx_ev, pos] = gap
    last = np.full(n, 1e7); m = lens > 0
    lt = E['h_ts'].list.last().to_numpy(); last[m] = (ts[m] - lt[m]) / 3600
    D = np.stack([np.log1p(E['h'].list.len().to_numpy()), np.log1p(np.clip(last, 0, 1e4)) / 5, (lens == 0).astype(float)], 1).astype(np.float32)
    return E['eid'], torch.from_numpy(H), torch.from_numpy(G), torch.from_numpy(D)
def row_arrays(E_eid, R):
    R = R.join(pl.DataFrame({'eid': E_eid, 'ei': np.arange(len(E_eid))}), on='eid', how='left').sort('ei', maintain_order=True)
    ei = R['ei'].to_numpy(); c = vid[R['ii'].to_numpy()]
    # slate: 10 candidates per event
    S = np.zeros((len(E_eid), 10), np.int64); k = np.arange(len(R)) - np.searchsorted(ei, ei); S[ei, np.minimum(k, 9)] = c
    return R, torch.from_numpy(ei), torch.from_numpy(c), torch.from_numpy(S)
Pe, PH, PG, PD = ev_arrays(P_ev); PRr, Pei, Pc, PS = row_arrays(Pe, P_rows); Py = torch.from_numpy(PRr['clicked'].to_numpy().astype(np.float32))
Te, TH, TG, TD = ev_arrays(T_ev); TRr, Tei, Tc, TS = row_arrays(Te, T_rows)
log('arrays built')
class DIN(nn.Module):
    def __init__(s, V, d):
        super().__init__(); s.E = nn.Embedding(V, d, padding_idx=0); s.B = nn.Embedding(V, 1); s.T = nn.Embedding(22, d, padding_idx=0)
        nn.init.normal_(s.E.weight, 0, 0.05); nn.init.zeros_(s.B.weight)
        s.att = nn.Sequential(nn.Linear(4 * d, 64), nn.ReLU(), nn.Linear(64, 1))
        s.mlp = nn.Sequential(nn.Linear(7 * d + 3, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
    def forward(s, H, G, D, c, S):
        m = (H > 0).float(); e = s.E(H) + s.T(G); ce = s.E(c); ce1 = ce.unsqueeze(1).expand_as(e)
        w = s.att(torch.cat([e, ce1, e * ce1, e - ce1], -1)).squeeze(-1).masked_fill(m == 0, -1e9)
        w = torch.softmax(w, 1) * (m.sum(1, keepdim=True) > 0).float()
        at = (w.unsqueeze(-1) * e).sum(1); mean = (e * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True).clamp(min=1)
        sl = s.E(S).mean(1)
        x = torch.cat([ce, at, mean, ce * at, ce * mean, ce - sl, sl, D], 1)
        return s.mlp(x).squeeze(-1) + s.B(c).squeeze(-1)
model = DIN(V, a.dim); opt = torch.optim.Adam(model.parameters(), lr=2e-3)
def batch(idx, ei, c, H, G, D, S): e = ei[idx]; return H[e], G[e], D[e], c[idx], S[e]
def predict(ei, c, H, G, D, S):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(c), 16384):
            idx = torch.arange(i, min(i + 16384, len(c))); out.append(model(*batch(idx, ei, c, H, G, D, S)))
    model.train(); return torch.cat(out).numpy()
n = len(Pc); lossf = nn.BCEWithLogitsLoss()
for ep in range(a.epochs):
    perm = torch.randperm(n); tl = 0
    for g in opt.param_groups: g['lr'] = 2e-3 * (0.4 ** ep)
    for bi, i in enumerate(range(0, n, a.bs)):
        idx = perm[i:i + a.bs]; p = model(*batch(idx, Pei, Pc, PH, PG, PD, PS)); loss = lossf(p, Py[idx])
        opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item()
    log(f'epoch {ep} loss {tl/(bi+1):.4f}')
    p = predict(Tei, Tc, TH, TG, TD, TS)
    if a.cut != 27: log(f'epoch {ep} target AUC {roc_auc_score(TRr["clicked"].to_numpy(), p):.5f}')
pl.DataFrame({'rowid': TRr['rowid'], 'seq_logit': p}).write_parquet(f'/workspace/huawei/v2work/ens/seq_c{a.cut}.parquet')
log('done total minutes', round((time.time() - t0) / 60, 2))
