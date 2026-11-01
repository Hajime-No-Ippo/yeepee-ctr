import numpy as np
from lib import *
from binormal import dp,auc
A=np.load('A_all.npy'); D=np.load('D_dcn.npy')
sets={'EN':EN,'E0':E0,'A':A,'D':D}; ks=list(sets); X=np.c_[[sets[k] for k in ks]].T
idx=np.random.RandomState(0).choice(len(EN),200000,replace=False); R0=np.corrcoef(X[idx].T)
rs=np.random.RandomState(1)
def pred(w,a,R):
    d=np.array([dp(x) for x in a]); return auc(w@d/np.sqrt(w@R@w))
probes={'P1':np.array([1,-.136,.098,0]),'Q1':np.array([1,-.136,.049,.049]),'Q2':np.array([1,-.136,0,.098])}
N=4000; out={k:[] for k in probes}
for i in range(N):
    aA=rs.uniform(.8185,.8205); aD=aA-rs.uniform(0,.0025); aE0=.824558; aEN=.828856
    R=R0.copy(); 
    for (i1,i2) in ((0,2),(0,3),(1,2),(1,3),(2,3)):
        v=np.clip(R[i1,i2]+rs.normal(0,.003),0,.9999); R[i1,i2]=R[i2,i1]=v
    if np.linalg.eigvalsh(R).min()<=0: continue
    err=rs.normal(0,.00005)  # binormal-model misspecification (calibration miss .00005-.0001)
    for k,w in probes.items(): out[k].append(pred(w,[aEN,aE0,aA,aD],R)-aEN+err)
for k,v in out.items():
    v=np.array(v); print(k,'mean %.6f  5%%/50%%/95%% %.6f %.6f %.6f  P(gain>0)=%.2f'%(v.mean(),*np.percentile(v,[5,50,95]),(v>0).mean()))
