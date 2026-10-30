"""avg_sq.py OUT DIR1 DIR2 ... : average seq-model outputs across seed runs; event-relative columns recomputed from the averaged logit.
Rows are stored event-major (10 consecutive rows per event) in every dir; rowid order is asserted identical."""
import sys, os, numpy as np, polars as pl
out, dirs = sys.argv[1], sys.argv[2:]; os.makedirs(out, exist_ok=True)
names = sorted(set.intersection(*[set(f[:-8] for f in os.listdir(d) if f.endswith('.parquet')) for d in dirs]))
for n in names:
    T = [pl.read_parquet(f'{d}/{n}.parquet') for d in dirs]; rid = T[0]['rowid'].to_numpy()
    for t in T[1:]: assert (t['rowid'].to_numpy() == rid).all(), (n, 'order')
    cols = [c for c in T[0].columns if c != 'rowid']
    M = {c: np.mean([t[c].to_numpy().astype(np.float64) for t in T], 0) for c in cols}
    S = M['sq_logit'].reshape(-1, 10)
    M['sq_z'] = ((S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-6)).ravel(); M['sq_dm'] = (S - S.mean(1, keepdims=True)).ravel()
    M['sq_lse'] = np.log(np.exp(S).sum(1, keepdims=True)).repeat(10, 1).ravel()
    M['sq_rank'] = (np.argsort(np.argsort(-S, 1, kind='stable'), 1) + 1).ravel(); M['sq_gap'] = (S - S.max(1, keepdims=True)).ravel()
    M['sq_gap2'] = (S - np.sort(S, 1)[:, -2:-1]).ravel()
    M['sq_sd'] = np.std([t['sq_logit'].to_numpy() for t in T], 0)   # seed disagreement
    df = pl.DataFrame({'rowid': rid, **M}).with_columns(pl.exclude('rowid').cast(pl.Float32)); df.write_parquet(f'{out}/{n}.parquet'); print(n, df.shape, flush=True)
