import json, numpy as np, polars as pl
from sklearn.metrics import roc_auc_score
V8 = '/workspace/huawei/v8'
for c in (25, 26, 23):
    v = pl.read_parquet(f'feat_v3/val{c}.parquet', columns=['rowid', 'clicked'])
    v0 = pl.read_parquet(f'{V8}/feat_v3/val{c}.parquet', columns=['rowid'])
    assert v['rowid'].equals(v0['rowid'])
    v = v.with_columns(pl.Series('z', np.load(f'ens/trim_blend_c{c}.npy')), pl.Series('zpp', np.load(f'{V8}/ens/pp_blend_c{c}.npy')))
    v = v.join(pl.read_parquet(f'{V8}/feat_sqv/all500/val{c}.parquet', columns=['rowid', 'sq_logit']), on='rowid', how='left', maintain_order='left')
    assert v['sq_logit'].null_count() == 0
    y = v['clicked'].to_numpy(); z = v['z'].to_numpy(); s = v['sq_logit'].to_numpy(); zp = v['zpp'].to_numpy()
    r = dict(cut=c, trim=round(roc_auc_score(y, z), 5), pp=round(roc_auc_score(y, zp), 5), v8=round(roc_auc_score(y, 0.8 * z + 0.2 * s), 5),
             pp_sq=round(roc_auc_score(y, 0.8 * zp + 0.2 * s), 5), corr_trim_pp=round(float(np.corrcoef(z, zp)[0, 1]), 5))
    print(json.dumps(r)); open('final_results.jsonl', 'a').write(json.dumps(r) + '\n')
