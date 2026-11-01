import numpy as np
from lib import *
from binormal import dp,auc
from scipy.optimize import minimize
A=np.load('A_all.npy'); D=np.load('D_dcn.npy')
sets={'EN':EN,'E0':E0,'A':A,'D':D}
ks=list(sets); X=np.c_[[sets[k] for k in ks]].T
idx=np.random.RandomState(0).choice(len(EN),200000,replace=False); R=np.corrcoef(X[idx].T); print(ks);print(R.round(4))
import sys
aD=float(sys.argv[1]) if len(sys.argv)>1 else .8192
a0={'EN':.828856,'E0':.824558,'A':.8197,'D':aD}
def pred(w,use):
    d=np.array([dp(a0[k]) for k in use]); ii=[ks.index(k) for k in use]; Rs=R[np.ix_(ii,ii)]; return auc(w@d/np.sqrt(w@Rs@w))
for use in (['EN','A'],['EN','D'],['EN','A','D'],['EN','E0','A'],['EN','E0','A','D']):
    r=minimize(lambda v:-pred(np.r_[1,v],use),np.zeros(len(use)-1),method='Nelder-Mead'); w=np.r_[1,r.x]
    print(use,'w',w.round(3),'gain',round(-r.fun-.828856,6))
print('--- constrained nonneg: EN - c*E0 + g*(1-q)A + g*q*D')
for aD in (.8192,.8180):
    a0['D']=aD
    for c in (.136,):
        for g in (.098,.15):
            for q in (0,.5,1):
                w=np.array([1,-c,g*(1-q),g*q]); print('aD',aD,'c',c,'g',g,'q',q,round(pred(w,ks)-.828856,6))
