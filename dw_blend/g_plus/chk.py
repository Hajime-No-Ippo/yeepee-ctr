import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[k]='1'
import numpy as np
from scipy.stats import norm, rankdata
from sklearn.metrics import roc_auc_score as AUC
H='/workspace/huawei/'; rk=lambda x:(rankdata(x)-1)/(len(x)-1)
pi=.0631
for c in (25,23):
    z=np.load(f'{H}cb_blend/oof_c{c}.npz',allow_pickle=True); y=z['y']; M=rk(z['raw_M']); G=rk(np.load(f'{H}v21graph/out/graph_c{c}.npy'))
    X=np.column_stack([M,G]); Xc=X-X.mean(0); Cov=Xc.T@Xc/len(X); aM,aG=AUC(y,M),AUC(y,G)
    # calibrate kappa on M
    from scipy.optimize import brentq
    f=lambda k,w,a: norm.cdf(k*(w@(a-.5))/np.sqrt(w@Cov@w-pi*(1-pi)*(w@(a-.5))**2)/np.sqrt(2))
    a=np.array([aM,aG]); kap=brentq(lambda k:f(k,np.array([1,0.]),a)-aM,.5,3)
    print(f'C{c} kappa {kap:.3f} aM {aM:.4f} aG {aG:.4f} rho {np.corrcoef(M,G)[0,1]:.3f}')
    for w in (.045,.09,.15):
        pred=f(kap,np.array([1,w]),a)-aM; act=AUC(y,M+w*G)-aM
        print(f'   w={w}: model pred {pred*1e5:+.0f}e-5  actual {act*1e5:+.0f}e-5')
