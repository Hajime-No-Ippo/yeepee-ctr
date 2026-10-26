"""v4_lite final blend: equal-weight mean of 2 component logits on eval -> eval_pred.parquet (rowid, p).
base = mean raw score of 2 seeds at 400 rounds; slow = 1 seed at 500 rounds."""
import glob, numpy as np, polars as pl
from cl import matrix
COL = {'base': 1, 'slow': 2}; NS = {'base': 2, 'slow': 1}
_, m, _ = matrix('eval', 1.0)
c = {}
for t, ci in COL.items():
    fs = sorted(glob.glob(f'ens/{t}_c27_s*.npy')); assert len(fs) == NS[t], (t, fs)
    c[t] = np.mean([np.load(f)[:, ci] for f in fs], 0); assert len(c[t]) == m.height
z = np.mean(list(c.values()), 0); p = 1 / (1 + np.exp(-z))
print('corr base/slow', round(float(np.corrcoef(c['base'], c['slow'])[0, 1]), 4))
pl.DataFrame({'rowid': m['rowid'], 'p': p}).write_parquet('eval_pred.parquet'); print('wrote eval_pred.parquet', m.height)
