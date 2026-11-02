import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[k]='1'
import numpy as np, polars as pl
from sklearn.metrics import roc_auc_score as AUC
from scipy.stats import kendalltau
P=dict(np.load('/workspace/huawei/dw_blend/pool_c25.npz')); y=pl.read_parquet('/workspace/huawei/v12/feat_v3/val25.parquet',columns=['clicked'])['clicked'].to_numpy()
M=P['M']; z=P['z']; n=len(y); rng=np.random.default_rng(0); idx=rng.choice(n,100000,replace=False)
for a in (.9,.95,.8):
    s2=a*M+(1-a)*z; fr=(1-kendalltau(M[idx],s2[idx])[0])/2
    d=[];
    for b in range(40):
        m=rng.random(n)<.5
        d.append(AUC(y[m],s2[m])-AUC(y[m],M[m]))
    print(f'mix {a}: reordered {fr:.4f}; full diff {AUC(y,s2)-AUC(y,M):+.6f}; half-sample paired diff sd {np.std(d):.6f}  (single-file half-sample AUC sd ~ {np.std([AUC(y[m],M[m]) for m in [rng.random(n)<.5 for _ in range(15)]]):.5f})')
