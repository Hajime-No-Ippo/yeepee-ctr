import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
O='/workspace/huawei/v21graph2/out/'
for c,k in [(25,'val25'),(23,'val23')]:
    d=np.load(O+f'tab_{k}.npz'); g=np.load(O+f'graph2s3b_f6_c{c}.npy')
    y=d['y'].reshape(-1,10); ts=d['ts']; H=d['hts']; n=len(ts)
    ye=y.max(1); gs=g.reshape(-1,10)
    hb=((H%86400)//21600); valid=H>0
    eb=((ts%86400)//21600)
    share=((hb==eb[:,None])&valid).sum(1)/np.maximum(valid.sum(1),1)
    hl=valid.sum(1)
    sh=(share*hl+0.25*3)/(hl+3)
    # hour-level circular closeness
    eh=(ts%86400)/3600.; hh=(H%86400)/3600.
    dd=np.abs(hh-eh[:,None]); dd=np.minimum(dd,24-dd)
    close=((dd<3)&valid).sum(1)/np.maximum(hl,1)
    base=np.c_[gs.mean(1),np.log(np.exp(gs-gs.max(1,keepdims=True)).sum(1))+gs.max(1),gs.max(1),np.log1p(hl)]
    last=np.where(hl>0,ts-np.max(np.where(valid,H,0),1),1e7); lg=np.log1p(last/3600)
    def ev(X):
        m=np.arange(n)%2==0
        lr=LogisticRegression(C=1,max_iter=500).fit(X[m],ye[m]); a=roc_auc_score(ye[~m],lr.predict_proba(X[~m])[:,1])
        lr=LogisticRegression(C=1,max_iter=500).fit(X[~m],ye[~m]); b=roc_auc_score(ye[m],lr.predict_proba(X[m])[:,1]); return (a+b)/2
    print(c,'event-any-click AUC base',ev(base),'+gap',ev(np.c_[base,lg]),'+share',ev(np.c_[base,sh]),'+close',ev(np.c_[base,close]),'share alone',roc_auc_score(ye,sh),'close alone',roc_auc_score(ye,close),flush=True)
