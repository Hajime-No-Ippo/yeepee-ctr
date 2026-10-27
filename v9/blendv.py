import sys, json, numpy as np
from sklearn.metrics import roc_auc_score
import polars as pl
tag, c = sys.argv[1], int(sys.argv[2])
y = pl.read_parquet(f'feat_v3/val{c}.parquet', columns=['clicked'])['clicked'].to_numpy()
b = np.load(f'ens/{tag}_b_c{c}_s0.npy')[:, 1]; s = np.load(f'ens/{tag}_s_c{c}_s0.npy')[:, 2]
z = (b + s) / 2
r = dict(tag=tag, cut=c, base=round(roc_auc_score(y, b), 5), slow=round(roc_auc_score(y, s), 5), blend=round(roc_auc_score(y, z), 5))
np.save(f'ens/{tag}_blend_c{c}.npy', z.astype(np.float32))
print(json.dumps(r), flush=True); open('scr_results.jsonl', 'a').write(json.dumps(r) + '\n')
