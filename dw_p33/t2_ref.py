import polars as pl,numpy as np
from sklearn.metrics import roc_auc_score as A
H='/workspace/huawei/'
m=pl.read_parquet(H+'v10/feat_v3/val25.parquet',columns=['rowid','clicked'])
y=m['clicked'].to_numpy()
for d,cols in (('feat_lr2',['lr12_cos','lr16_cos','lr8_cos','lr12_con05']),('feat_pp',['x_pp_hcos','x_pp_con05_k32','x_pp_last3']),('feat_v3',['sv_k32_hcos','sv_k16_hcos','sv_k64_hcos','co_cos_rw','co_cos_max20','mf_score']),('feat_dw',['hist_cos_imp8','hist_cos_impspace','impsvd_cos'])):
    f=pl.read_parquet(H+f'v12/{d}/val25.parquet',columns=['rowid']+cols); j=m.select('rowid').join(f,on='rowid',how='left')
    for c in cols:
        x=j[c].fill_null(0).fill_nan(0).to_numpy(); print(d,c,round(A(y,x),4))
