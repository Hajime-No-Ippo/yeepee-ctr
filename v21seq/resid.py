import numpy as np, sys
from evalseq import *
def slate(v,c):
    y=yv(c).reshape(-1,10); z=zbase(v,c); Lg,_=listp(v,c); z2=.4*z+.6*Lg
    s=np.load(f'seqfeat/{v}_val{c}.npz',allow_pickle=True); return y,Lg,z2,s
for c in (25,23):
    for v in 'AB':
        y,Lg,z2,s=slate(v,c); n=len(y); hl=s['hlen'].astype(int); g0=s['hgap'][:,0]
        # slate-level residual: mean y - mean p, p = softmax with outside option approximated by exp(listlp)
        lp=np.load(f'/workspace/huawei/v21diag/ens/v21list{v}_c{c}_s0.npy')[:,3].reshape(-1,10); p=np.exp(lp)
        r=y.mean(1)-p.mean(1); print(f'== C{c} {v}: slates {n} mean y {y.mean():.4f} mean p {p.mean():.4f}  list slate-AUC-level mean resid {r.mean():+.5f}')
        def rep(name,m): print(f'  {name:28s} n={m.sum():6d} resid {r[m].mean():+.5f}  (rel to overall {r[m].mean()-r.mean():+.5f})  y {y[m].mean():.4f}')
        for nm,m in [('hlen<=6',(hl<=6)&(hl>0)),('hlen 0',hl==0),('hlen 7-18',(hl>=7)&(hl<=18)),('hlen 19-20',hl>=19)]: rep(nm,m)
        has=hl>0; qs=np.quantile(g0[has],[.2,.4,.6,.8])
        b=np.where(has,np.digitize(g0,qs),-1)
        for k in range(5): rep(f'stale quintile {k} (fresh=0)',b==k)
