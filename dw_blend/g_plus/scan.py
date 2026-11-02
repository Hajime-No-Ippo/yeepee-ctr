import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[k]='1'
import numpy as np, polars as pl
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score as AUC
H='/workspace/huawei/'; rk=lambda x:((rankdata(x)-1)/(len(x)-1))
res={}
for c in (23,25,26):
    z=np.load(f'{H}cb_blend/oof_c{c}.npz',allow_pickle=True); y=z['y']
    G=rk(np.load(f'{H}v21graph/out/graph_c{c}.npy').astype(np.float64))
    print(c,'keys',[k for k in z.files if k.startswith('raw_')],'n',len(y),'graphAUC',round(AUC(y,G),5),flush=True)
    R={k[4:]:rk(z[k].astype(np.float64)) for k in z.files if k.startswith('raw_')}
    bases={}
    if 'z' in R and 'M' in R: bases['.4z+.6M']=.4*R['z']+.6*R['M']
    if all(k in R for k in ('z','M','D')): bases['.3z+.3M+.4D']=.3*R['z']+.3*R['M']+.4*R['D']
    if 'M' in R: bases['M']=R['M']
    for bn,b in bases.items():
        b=rk(b); a0=AUC(y,b)
        for w in (.03,.045,.06,.075,.09,.12,.15,.2):   # additive: final = rank(b)+w*rank(G)
            res[(c,bn,w)]=AUC(y,b+w*G)-a0
        print(c,bn,'base',round(a0,5),' '.join(f'w{w}:{res[(c,bn,w)]*1e5:+.0f}' for w in (.03,.045,.06,.075,.09,.12,.15,.2)),flush=True)
