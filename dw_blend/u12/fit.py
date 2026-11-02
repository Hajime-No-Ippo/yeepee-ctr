import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[k]='1'
import numpy as np, polars as pl
from scipy.stats import norm
Z=np.load('arrs.npz'); V='/workspace/huawei/v21fit/'
V8=np.load(V+'inv_v8regen.npy'); V5=np.load(V+'inv_v5.npy')
C={'EN':Z['EN'],'E0':Z['E0'],'A':Z['A'],'D':Z['D'],'V8':V8,'V5':V5}
names=list(C); X=np.column_stack([C[k] for k in names]).astype(np.float64)
pi=np.mean([pl.read_parquet(f'/workspace/huawei/v12/feat_v3/val{c}.parquet',columns=['clicked'])['clicked'].mean() for c in (23,25)])
print('pi',pi)
mu=X.mean(0); Xc=X-mu; Cov=Xc.T@Xc/len(X)    # covariance of uniform ranks
print(np.round(np.corrcoef(X.T),4))
# blend weights
W={'EN':[1,0,0,0,0,0],
 'Q2':[1,-.136,0,.098,0,0],'R1':[1,-.136,.049,.049,-.03,-.015]}
def w(c,g,q,v8,v5): return np.array([1,-c,g*(1-q),g*q,v8,v5])
W={'EN':w(0,0,0,0,0)*0+np.array([1,0,0,0,0,0.]),
 'Q2':np.array([1,-.136,0,.098,0,0]),'R1':np.array([1,-.136,.049,.049,-.03,-.015]),
 'S1':w(.10902810947726392,.12251977564719992,.665,-.052490334215038625,-.033138516122339604),
 'T1':w(.081,.148,.834,-.075,-.052),'U1':w(.09,.16,1.0,-.09,-.04),'U2':w(.09,.16,.9,-.09,-.04)}
pub={'EN':.828856,'Q2':.829016,'R1':.829058,'S1':.829101,'T1':.829096}
def auc(a,wt,kappa=1.0):
    # a = AUCs of components (EN,E0,A,D,V8,V5); mean shift in uniform-rank units
    delta=wt@(a-.5); var=wt@Cov@wt; sw2=var-pi*(1-pi)*delta**2
    return norm.cdf(kappa*delta/np.sqrt(sw2)/np.sqrt(2))
a0=np.array([.828856,.824558,.8194,.8181,.8146,.7925])
print('EN model auc (kappa=1)',auc(a0,W['EN']),'E0',auc(a0,np.array([0,1,0,0,0,0.])))
# calibrate kappa on single-component: EN. (E0 as check)
from scipy.optimize import brentq
kap=brentq(lambda k:auc(a0,W['EN'],k)-.828856,.5,3); print('kappa',kap,'E0 check',auc(a0,np.array([0,1.,0,0,0,0]),kap),'(true .824558)')
# Bayesian: unknown aA,aD,aV8,aV5 ; priors from local; shared bias b sd .00005; noise sd sig
rng=np.random.default_rng(1)
def run(sig,prior_sd=.004,N=400000):
    pm=np.array([.8194,.8181,.8146,.7925])
    S=pm+rng.normal(size=(N,4))*np.array([.004,.004,.006,.008])*prior_sd/.004
    b=rng.normal(size=N)*.00005
    keys=['EN','Q2','R1','S1','T1','U1','U2']
    out={k:np.empty(N) for k in keys}
    for i in range(N):
        a=np.array([.828856,.824558,*S[i]])
        for k in keys: out[k][i]=auc(a,W[k],kap)
    # likelihood: pub - b = model + noise
    ll=np.zeros(N)
    for k,v in pub.items(): ll+=-0.5*((v-b-out[k])/sig)**2
    wt=np.exp(ll-ll.max()); wt/=wt.sum(); ess=1/np.sum(wt**2)
    return out,b,wt,ess,S
def wq(x,wt,q):
    o=np.argsort(x); c=np.cumsum(wt[o]); return np.interp(q,c,x[o])
for sig in (.00004,.00005,.00006):
    out,b,wt,ess,S=run(sig,N=60000)
    print(f'\nsig {sig} ESS {ess:.0f}  post means aA,aD,aV8,aV5',np.round(wt@S,4))
    for k in ('U1','U2','T1'):
        pred=out[k]+b   # public-scale prediction (incl shared bias)
        d=pred-(out['S1']+b)
        # add measurement noise for observed U1 vs S1 diff (independent noise on both): sd sig*sqrt2
        dn=d+np.random.default_rng(2).normal(size=len(d))*sig*np.sqrt(2)
        print(f' {k}: pred public {wt@pred:.6f} [90% {wq(pred,wt,.05):.6f},{wq(pred,wt,.95):.6f}]  gain vs S1 {wt@d*1e5:+.2f}e-5 [{wq(d,wt,.05)*1e5:+.1f},{wq(d,wt,.95)*1e5:+.1f}] P(true>S1)={wt@(d>0):.2f}  P(observed>S1 | noise)={wt@(dn>0):.2f}  P(gain>=+5e-5)={wt@(d>=5e-5):.2f}')
