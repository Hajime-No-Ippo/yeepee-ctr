"""S3 blend tweak: final logit = z + w * within-event standardised z (monotone within event), output sigmoid. python blendtweak.py TAG"""
import sys, json, numpy as np, polars as pl
from sklearn.metrics import roc_auc_score
tag = sys.argv[1]; res = {}
for c in (25, 26, 23):
    try: z = np.load(f'ens/{tag}_blend_c{c}.npy').astype(np.float64)
    except FileNotFoundError: continue
    m = pl.read_parquet(f'feat_v3/val{c}.parquet', columns=['eid', 'clicked']); y = m['clicked'].to_numpy()
    d = m.select('eid').with_columns(pl.Series('z', z)); zz = d.select(((pl.col('z') - pl.col('z').mean().over('eid')) / (pl.col('z').std().over('eid').fill_null(1) + 1e-6)).fill_nan(0))['z'].to_numpy()
    dm = d.select(pl.col('z') - pl.col('z').max().over('eid'))['z'].to_numpy()
    res[c] = {f'z{w}': round(roc_auc_score(y, z + w * zz), 5) for w in (-0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.5)}
    res[c].update({f'dm{w}': round(roc_auc_score(y, z + w * dm), 5) for w in (0.1, 0.2, 0.4)})
    print(c, json.dumps(res[c]), flush=True)
