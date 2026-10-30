import os; os.environ['OMP_NUM_THREADS']='2'
import sys, numpy as np
sys.path.insert(0,'.'); from diag1 import *; from combine3 import get, comb
from sklearn.metrics import roc_auc_score
for v in 'BA':
    y,Z=get(25,v,'blend'); P=np.load(f'ens/pslate_c25_{v}.npy')[:,0]; s0=Z.ravel(); s1=comb(Z,P,.3,.5); E=len(P); rng=np.random.RandomState(1)
    print(v,'base',round(AUC(y,s0),5),'new',round(AUC(y,s1),5))
    print(' grid delta (a rows; c cols 0,.25,.5,.75,1):')
    for a in (.1,.2,.3,.4,.5): print('  a',a,[round(AUC(y,comb(Z,P,a,c).ravel())-AUC(y,s0),5) for c in (0,.25,.5,.75,1)])
    Y=y.reshape(E,10); S0=s0.reshape(E,10); S1=s1.reshape(E,10); d=[]
    for r in range(40):
        ix=rng.randint(0,E,E); yy=Y[ix].ravel(); d.append(AUC(yy,S1[ix].ravel())-AUC(yy,S0[ix].ravel()))
    print(' event-bootstrap delta mean %.5f sd %.5f  P(delta>0)=%.2f'%(np.mean(d),np.std(d),np.mean(np.array(d)>0)))
    meta=load_meta(25); st=meta['tm_stale'].to_numpy(); 
    for lo,hi,n in ((0,86400,'<1d'),(86400,1e12,'>=1d')):
        k=(st>=lo)&(st<hi); print(' ',n,'delta',round(AUC(y[k],s1[k])-AUC(y[k],s0[k]),5))
    h=meta['us_nohist'].to_numpy()==1; k=~h; print('  with-history delta',round(AUC(y[k],s1[k])-AUC(y[k],s0[k]),5),' empty-hist delta',round(AUC(y[h],s1[h])-AUC(y[h],s0[h]),5))
