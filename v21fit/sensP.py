import numpy as np
from lib import *
from binormal import dp,auc
from scipy.optimize import minimize
A=np.load('A_all.npy'); D=np.load('D_dcn.npy')
sets={'EN':EN,'E0':E0,'A':A,'D':D,'P':P}; ks=list(sets); X=np.c_[[sets[k] for k in ks]].T
idx=np.random.RandomState(0).choice(len(EN),200000,replace=False); R=np.corrcoef(X[idx].T); print(ks); print(R.round(4))
def pred(w,a): d=np.array([dp(x) for x in a]); return auc(w@d/np.sqrt(w@R@w))
base=np.array([1,-.136,.049,.049,0])
for aP in (.800,.8052,.810):
    a=[.828856,.824558,.8197,.8192,aP]
    r=minimize(lambda v:-pred(np.r_[base[:4],v[0]],a),[0],method='Nelder-Mead')
    print('aP',aP,'opt wP',round(r.x[0],3),'gain vs Q1-form',round(-r.fun-pred(base,a),6),' Q1 itself',round(pred(base,a)-.828856,6))
    for wP in (-.02,-.04): print('   wP',wP,round(pred(np.r_[base[:4],wP],a)-.828856,6))
