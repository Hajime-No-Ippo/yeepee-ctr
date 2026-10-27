"""Incremental-value harness. Stack on validation rows: GroupKFold by user (5 folds); LGB on [v11 logit + within-event transforms + flags] (BASE) vs BASE + candidate features.
Reports OOF global AUC, 48h+/24h+ slices, segments, paired event-bootstrap CI of delta."""
import numpy as np, lightgbm as lgb
from sklearn.metrics import roc_auc_score as A
from sklearn.model_selection import GroupKFold
from common import *
def base_frame(D,V,c):
    z=v11_pred(c); zz,dm,rk=within_event(z); Z=z.reshape(-1,10); n=len(Z)
    sg=segs(c); hl=np.repeat(D.Hlen[V['e'][::10]],10)
    F={'v11':z,'v11_z':zz,'v11_dm':dm,'v11_rk':rk,'ev_lse':np.repeat(np.log(np.exp(Z).sum(1)),10),'ev_max':np.repeat(Z.max(1),10),'ev_mean':np.repeat(Z.mean(1),10),'hist_len':hl.astype(np.float32),'nohist':sg['nohist'].astype(np.float32)}
    return F
PARAMS=dict(objective='binary',learning_rate=0.05,num_leaves=31,min_data_in_leaf=300,feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10,max_bin=127,verbose=-1,num_threads=4)
def oof(X,y,groups,seed,rounds=250,params=PARAMS):
    p=np.zeros(len(y))
    for tr,te in GroupKFold(5).split(X,y,groups):
        d=lgb.Dataset(X[tr],y[tr]); m=lgb.train({**params,'seed':seed},d,rounds); p[te]=m.predict(X[te],raw_score=True)
    return p
def boot(y,p0,p1,ev,nb=100,seed=0,mask=None):
    rng=np.random.default_rng(seed); ne=ev.max()+1; d=[]
    for _ in range(nb):
        w=np.bincount(rng.integers(0,ne,ne),minlength=ne)[ev].astype(float)
        if mask is not None: w=w*mask
        d.append(A(y,p1,sample_weight=w)-A(y,p0,sample_weight=w))
    return np.percentile(d,[2.5,97.5])
def evaluate(F_base,F_new,y,user,stale,ev,seeds=(0,1),rounds=250,nohist=None):
    Xb=np.column_stack(list(F_base.values())).astype(np.float32); Xn=np.column_stack(list(F_base.values())+list(F_new.values())).astype(np.float32)
    p0=np.mean([oof(Xb,y,user,s,rounds) for s in seeds],0); p1=np.mean([oof(Xn,y,user,s,rounds) for s in seeds],0)
    r=dict(base=A(y,p0),new=A(y,p1)); r['delta']=r['new']-r['base']; r['ci']=tuple(boot(y,p0,p1,ev))
    for nm,m in (('stale24p',stale>=24),('stale48p',stale>=48),('nohist',nohist==1),('hist',nohist==0)):
        if m.sum()>3000 and 0<y[m].sum(): r['d_'+nm]=A(y[m],p1[m])-A(y[m],p0[m]); r['n_'+nm]=int(m.sum())
    return r,p0,p1

PI=dict(objective='binary',learning_rate=0.03,num_leaves=15,min_data_in_leaf=500,feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,lambda_l2=30,max_bin=127,verbose=-1,num_threads=4)
def oof_init(X,y,groups,init,seed,rounds=200):
    """residual boosting on top of the v11 logit (init_score): the base can't lose v11 information"""
    p=np.zeros(len(y))
    for tr,te in GroupKFold(5).split(X,y,groups):
        d=lgb.Dataset(X[tr],y[tr],init_score=init[tr]); m=lgb.train({**PI,'seed':seed},d,rounds); p[te]=m.predict(X[te],raw_score=True)+init[te]
    return p
