import sys, numpy as np, polars as pl, json
from sklearn.metrics import roc_auc_score
dirs = sys.argv[1].split(','); 
for c in (25, 26):
    v = pl.read_parquet(f'feat_v3/val{c}.parquet', columns=['rowid','clicked']).with_columns(pl.Series('z', np.load(f'ens/pp_blend_c{c}.npy')))
    for k, d in enumerate(dirs): v = v.join(pl.read_parquet(f'{d}/val{c}.parquet', columns=['rowid','sq_logit']).rename({'sq_logit': f's{k}'}), on='rowid', how='inner')
    y = v['clicked'].to_numpy(); z = v['z'].to_numpy(); s = np.mean([v[f's{k}'].to_numpy() for k in range(len(dirs))], 0)
    r = dict(ens=sys.argv[1], cut=c, sq=round(roc_auc_score(y, s), 5))
    for w in (0.1, 0.2, 0.3, 0.4): r[f'raw{w}'] = round(roc_auc_score(y, (1-w)*z + w*s), 5)
    print(json.dumps(r))
