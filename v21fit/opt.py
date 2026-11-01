import numpy as np
from lib import *
from binormal import dp,auc
from scipy.optimize import minimize
A=np.load('A_all.npy'); B=np.load('B_all.npy')
sets={'EN':EN,'E0':E0,'O':O,'P':P,'A':A,'B':B}
ks=list(sets); X=np.c_[[sets[k] for k in ks]].T
idx=np.random.RandomState(0).choice(len(EN),200000,replace=False)
R=np.corrcoef(X[idx].T)

print(ks);print(R.round(4))
a0={'EN':.828856,'E0':.824558,'O':.8190,'P':.8052,'A':.8197,'B':.8195}
def pred(w,use):
    d=np.array([dp(a0[k]) for k in use]); Rs=R[np.ix_([ks.index(k) for k in use],[ks.index(k) for k in use])]
    return auc(w@d/np.sqrt(w@Rs@w))
for use in (['EN','A'],['EN','O'],['EN','E0','A'],['EN','E0','O'],['EN','E0','A','P']):
    f=lambda v:-pred(np.r_[1,v],use)
    r=minimize(f,np.zeros(len(use)-1),method='Nelder-Mead'); w=np.r_[1,r.x]
    print(use,'opt w',w.round(3),'gain',round(-r.fun-.828856,6))
    # constrained: nonneg for non-E0
for wA in (.05,.08,.1,.15): print('EN+A w',wA,round(pred(np.array([1-wA,wA]),['EN','A'])-.828856,6))
