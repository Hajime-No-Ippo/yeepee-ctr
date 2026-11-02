import os
os.environ['OMP_NUM_THREADS']='2'
import numpy as np, polars as pl
from sklearn.metrics import roc_auc_score as AUC
H='/workspace/huawei'
for c in (23,25):
    z=np.load(f'{H}/cb_blend/oof_c{c}.npz',allow_pickle=True); y=z['y']
    B=np.mean([z['raw_'+n].astype(np.float64) for n in ('z','M','big','D')],0)
    e=pl.read_parquet(f'{H}/v12/feat_v3/val{c}.parquet',columns=['eid'])['eid']
    d=pl.DataFrame({'e':e,'b':B}).with_columns(pl.col('b').mean().over('e').alias('m'),pl.col('b').exp().sum().over('e').log().alias('lse'),pl.col('b').max().over('e').alias('mx'))
    m=d['m'].to_numpy();lse=d['lse'].to_numpy();mx=d['mx'].to_numpy()
    print(c,'base',round(AUC(y,B),5))
    for lam in (-0.2,-0.1,0.1,0.2,0.4):
        print(c,'lam',lam,'B+lam*(B-evmean)',round(AUC(y,B+lam*(B-m)),5),'B+lam*(B-lse)',round(AUC(y,B+lam*(B-lse)),5),'B+lam*(B-evmax)',round(AUC(y,B+lam*(B-mx)),5),flush=True)
