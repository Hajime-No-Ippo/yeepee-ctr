"""v4 final blend: equal-weight mean of component logits (each = seed-mean raw score) on eval -> eval_pred_v4.parquet (rowid, p)."""
import glob, numpy as np, polars as pl
from cl import matrix
COL = {'base': 1, 'wide': 1, 'slow': 2, 'estrc': 1}; NS = {'base': 7, 'wide': 2, 'slow': 1, 'estrc': 2}
_, m, _ = matrix('eval', 1.0); _, m2, _ = matrix('eval', 1.0, variant='stale', extra=('feat_rc',))
assert (m['rowid'] == m2['rowid']).all()
c = {}
for t, ci in COL.items():
    fs = sorted(glob.glob(f'ens/{t}_c27_s*.npy')); assert len(fs) == NS[t], (t, fs)
    c[t] = np.mean([np.load(f)[:, ci] for f in fs], 0); assert len(c[t]) == m.height
z = np.mean(list(c.values()), 0); p = 1 / (1 + np.exp(-z))
print('corr with base', {k: round(float(np.corrcoef(v, c['base'])[0, 1]), 4) for k, v in c.items()})
v3 = m.select('rowid').join(pl.read_parquet('eval_pred_v3.parquet'), on='rowid', how='left')['p'].to_numpy()
b3 = np.mean([np.load(f'ens/base_c27_s{i}.npy')[:, 1] for i in range(3)], 0)
print('v3 vs ens base s0-2 corr', round(float(np.corrcoef(1/(1+np.exp(-b3)), v3)[0, 1]), 5), 'v3 vs v4 corr', round(float(np.corrcoef(p, v3)[0, 1]), 4))
pl.DataFrame({'rowid': m['rowid'], 'p': p}).write_parquet('eval_pred_v4.parquet'); print('wrote eval_pred_v4.parquet', m.height)
