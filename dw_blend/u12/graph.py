import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[k]='1'
import numpy as np
from scipy.stats import norm, rankdata, spearmanr
Z=np.load('arrs.npz'); G0=np.load('/workspace/huawei/v21graph/out/graph_eval_merged_rank.npy'); print(G0.shape,G0.dtype,np.isnan(G0).sum())
G=(rankdata(G0)-1)/(len(G0)-1)
import hashlib
for k in ('EN','A','D','S1','U1','E0'): print('rho graph vs',k,round(spearmanr(G[::5],Z[k][::5])[0],4))
# check merged order sanity: compare graph v3-order file mapped? skip; use rho EN as sanity (nonzero strong)
pi=.0631
X=np.column_stack([Z[k] for k in ('S1','U1')]+[G]).astype(float); Cov=np.cov(X.T)
a_S1=.829101; 
def auc(wt,aS,aG,kap=1.1328):
    a=np.array([aS,aS,aG]); delta=wt@(a-.5); var=wt@Cov@wt; sw2=var-pi*(1-pi)*delta**2
    return norm.cdf(kap*delta/np.sqrt(sw2)/np.sqrt(2))
for aG in (.79,.795,.80,.805,.81):
    base=auc(np.array([1,0,0.]),a_S1,aG)
    r=[(w,(auc(np.array([1,0,w]),a_S1,aG)-base)*1e5) for w in (-.09,-.045,.045,.09,.15)]
    print('graph AUC',aG,' gain over S1 (e-5) by weight:',[(w,round(g,1)) for w,g in r])
