"""v10 final blend (no sequence post-blend: the sequence model is stacked into both GBDTs via feat_sq10)
v8 final blend on eval -> eval_pred.parquet (rowid, p).
z = mean of 2 GBDT component logits (base = 1 seed at 400 rounds, slow = 1 seed at 500 rounds);
z8 = 0.8 * z + 0.2 * sq_logit (seq.py sequence model, raw logit), p = sigmoid(z8) in [0, 1]."""
import glob, numpy as np, polars as pl
from cl import matrix
COL = {'base': 1, 'slow': 2}; NS = {'base': 1, 'slow': 1}; WSQ = 0.0
_, m, _ = matrix('eval', 1.0)
c = {}
for t, ci in COL.items():
    fs = sorted(glob.glob(f'ens/{t}_c27_s*.npy')); assert len(fs) == NS[t], (t, fs)
    c[t] = np.mean([np.load(f)[:, ci] for f in fs], 0); assert len(c[t]) == m.height
z = np.mean(list(c.values()), 0)
sq = pl.read_parquet('feat_sq10/eval.parquet', columns=['rowid', 'sq_logit'])
j = m.select('rowid').join(sq.with_columns(pl.col('rowid').cast(m['rowid'].dtype)), on='rowid', how='left', maintain_order='left')
assert j.height == m.height and j['sq_logit'].null_count() == 0 and (j['rowid'] == m['rowid']).all()
s = j['sq_logit'].to_numpy().astype(np.float64)
z8 = (1 - WSQ) * z + WSQ * s; p = 1 / (1 + np.exp(-z8)); assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
print('corr base/slow', round(float(np.corrcoef(c['base'], c['slow'])[0, 1]), 4), 'corr z/sq', round(float(np.corrcoef(z, s)[0, 1]), 4))
pl.DataFrame({'rowid': m['rowid'], 'p': p}).write_parquet('eval_pred.parquet'); print('wrote eval_pred.parquet', m.height)
