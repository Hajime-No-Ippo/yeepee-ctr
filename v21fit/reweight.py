"""Re-weight a new model array on top of S1.  Usage (cores 6-7):
 taskset -c 6-7 /workspace/venv_v17/bin/python reweight.py --path /workspace/huawei/v21cat/out/cat_eval.npy [--col 0] [--order v3|merged] [--logit] \
    --gain25 0.0004 --w25 0.2 [--gain23 0.0006 --w23 0.2] [--rho25 0.97] [--bar 0.00005]
 gainXX = local gain over the cut's base when blended with weight wXX (base .4z+.6M: C25 .81921, C23 .79903).
 Nothing is built; prints weight (shrunk 50% toward 0), predicted gain over S1 with 90% range and P>0."""
import argparse,sys,numpy as np
ap=argparse.ArgumentParser(); ap.add_argument('--path',required=True); ap.add_argument('--col',type=int,default=None); ap.add_argument('--order',default='v3'); ap.add_argument('--logit',action='store_true')
ap.add_argument('--gain25',type=float); ap.add_argument('--w25',type=float,default=.2); ap.add_argument('--gain23',type=float); ap.add_argument('--w23',type=float,default=.2)
ap.add_argument('--rho25',type=float); ap.add_argument('--bar',type=float,default=.00005); ap.add_argument('--shrink',type=float,default=.5); ap.add_argument('--transfer_sd',type=float,default=.0008)
a=ap.parse_args()
import polars as pl
from lib import *
import addon
from addon import sets,ix,n,ks
from binormal import dp,auc,comb as bcomb
from scipy.optimize import brentq
x=np.load(a.path)
if x.ndim==2: x=x[:,a.col if a.col is not None else 0]
x=x.astype(np.float64); assert len(x)==529390 and np.isfinite(x).all()
if a.order=='v3':
    rid=pl.read_parquet('/workspace/huawei/v12/feat_v3/eval.parquet',columns=['rowid'])['rowid'].to_list()
    x=pl.DataFrame({'rowid':rid,'p':x}).join(pl.DataFrame({'rowid':M['rowid'].to_list()}),on='rowid',how='right',maintain_order='right')['p'].to_numpy()
X=nr(x)
idx=np.random.RandomState(0).choice(len(EN),200000,replace=False)
rho={k:np.corrcoef(X[idx],sets[k][idx])[0,1] for k in ks}
print('eval rho:',{k:round(v,4) for k,v in rho.items()})
# local AUC of the new model from its C25/C23 gain (binormal, rho to base ~ rho with A unless given)
base={'25':.81921,'23':.79903}; offs=[]
for cut,g,w,r in (('25',a.gain25,a.w25,a.rho25 if a.rho25 else rho['A']),('23',a.gain23,a.w23,rho['A'])):
    if g is None: continue
    f=lambda a2: bcomb(base[cut],a2,r,w)-base[cut]-g
    try: a2=brentq(f,.70,.90)
    except ValueError: print('cannot solve local AUC for cut',cut); sys.exit(1)
    offs.append(a2-base[cut]); print('cut C%s: gain %.5f at w %.2f, rho %.4f -> local AUC %.5f (offset vs base %+.5f)'%(cut,g,w,r,a2,a2-base[cut]))
off=np.mean(offs)
P=np.load('post5.npz'); dd,RR,aEN=P['d'],P['R'],P['a_EN']; K=len(dd); rs=np.random.RandomState(5)
pS1=np.load('pS1.npy'); c,g_,q,v8,v5=pS1
wS1=np.zeros(n); wS1[0]=1; wS1[1]=-c; wS1[ix['A']]=g_*(1-q); wS1[ix['D']]=g_*q; wS1[ix['v8r']]=v8; wS1[ix['v5']]=v5
# A's public AUC varies by draw: recover from draw (dd[:,A]); new public = A_pub + offset + transfer noise
aA=auc(dd[:,ix['A']]); anew=aA+off+rs.normal(0,a.transfer_sd,K); dnew=dp(anew)
m=n+1; D2=np.c_[dd,dnew]; R2=np.zeros((K,m,m)); R2[:,:n,:n]=RR
rv=np.array([rho[k] for k in ks])
for t in range(K):
    j=np.minimum(rv+rs.normal(0,1,n)*np.where(rv>.99,.0008,.003),.9999); R2[t,n,:n]=j; R2[t,:n,n]=j; R2[t,n,n]=1
def A_(w): return auc((D2@w)/np.sqrt(np.einsum('i,nij,j->n',w,R2,w)))
w0=np.r_[wS1,0]; base_=A_(w0)
grid=np.arange(-.15,.301,.01); mg=np.array([ (A_(w0+wt*np.eye(m)[n])-base_).mean() for wt in grid]); wo=grid[mg.argmax()]
print('new-model public AUC prior-> %.5f (5/95 %.5f %.5f)'%(anew.mean(),*np.percentile(anew,[5,95])))
print('optimum weight on S1 (rank space, unshrunk) %.3f, mean gain %+.6f'%(wo,mg.max()))
ws=wo*a.shrink; g=A_(w0+ws*np.eye(m)[n])-base_
print('SHRUNK weight %.3f: gain over S1 mean %+.6f 5/95 %+.6f %+.6f P>0 %.2f'%(ws,g.mean(),*np.percentile(g,[5,95]),(g>0).mean()))
ok=g.mean()>=a.bar and (g>0).mean()>.70
print('BAR (>= %.5f & P>0>.70): %s'%(a.bar,'PASS -> build probe: S1 + %.3f*new, then rank-scale'%ws if ok else 'FAIL -> do not build'))
