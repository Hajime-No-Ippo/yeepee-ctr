"""cmp.py TAG [TAG...]: AUC of z=(b400+s500)/2 and z8=0.8z+0.2sq vs trim (=v8 lite), plain and eval-likeness-weighted (w_advni)."""
import sys, json, numpy as np, polars as pl
from sklearn.metrics import roc_auc_score as A
V8='/workspace/huawei/v8'
for tag in sys.argv[1:]:
  for c in (25,26,23):
    try: b=np.load(f'ens/{tag}_b_c{c}_s0.npy')[:,1]; s=np.load(f'ens/{tag}_s_c{c}_s0.npy')[:,2]
    except FileNotFoundError: continue
    v=pl.read_parquet(f'feat_v3/val{c}.parquet',columns=['rowid','clicked']).join(pl.read_parquet(f'{V8}/feat_sqv/all500/val{c}.parquet',columns=['rowid','sq_logit']),on='rowid',how='left',maintain_order='left')
    y=v['clicked'].to_numpy(); sq=v['sq_logit'].to_numpy(); w=np.load(f'ens/w_advni_val{c}.npy')
    zt=(np.load(f'ens/trim_b_c{c}_s0.npy')[:,1]+np.load(f'ens/trim_s_c{c}_s0.npy')[:,2])/2; z=(b+s)/2
    r=dict(tag=tag,cut=c,z=round(A(y,z),5),dz=round(A(y,z)-A(y,zt),5),z8=round(A(y,.8*z+.2*sq),5),dz8=round(A(y,.8*z+.2*sq)-A(y,.8*zt+.2*sq),5),
           dz8_w=round(A(y,.8*z+.2*sq,sample_weight=w)-A(y,.8*zt+.2*sq,sample_weight=w),5))
    print(json.dumps(r),flush=True); open('cmp_results.jsonl','a').write(json.dumps(r)+'\n')
