import numpy as np
from evalseq import *
from sklearn.linear_model import LogisticRegression
def feats(v,c):
    s=np.load(f'seqfeat/{v}_val{c}.npz',allow_pickle=True); hl=s['hlen'].astype(int); g0=s['hgap'][:,0]; g=np.where(hl>0,np.log1p(np.maximum(g0,0)/3600),0)
    F=np.c_[hl==0,(hl>0)&(hl<=6),hl>=19,g,g**2,(hl>0)*np.clip(g,0,None)>np.log1p(24)].astype(float); return F
for v in 'AB':
    D={}
    for c in (25,23):
        y=yv(c); z=zbase(v,c); Lg,_=listp(v,c); z2=.4*z+.6*Lg; D[c]=(y.reshape(-1,10),z2.reshape(-1,10),feats(v,c))
    for tr,te in ((25,23),(23,25)):
        y,z2,F=D[tr]; # fit offset: logistic on row with base logit z2 as offset via iterative: use LR on [z2, F broadcast]
        Fr=np.repeat(F,10,0); X=np.c_[z2.reshape(-1),Fr]; m=LogisticRegression(C=1.0,max_iter=500).fit(X,y.reshape(-1))
        y2,zz,F2=D[te]; X2=np.c_[zz.reshape(-1),np.repeat(F2,10,0)]; s=m.decision_function(X2)
        print(v,f'fit C{tr} -> test C{te}: base {AUC(y2.reshape(-1),zz.reshape(-1)):.5f}  corrected {AUC(y2.reshape(-1),s):.5f}  coef {np.round(m.coef_[0],3)}')
