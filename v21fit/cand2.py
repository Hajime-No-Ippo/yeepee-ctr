import numpy as np, pickle
from lib import *
from binormal import dp,auc
import addon
from addon import n,ix,wQ,sets
draws,W,bm,bv=pickle.load(open('post.pkl','rb'))
new={'v5c':np.load('inv_v5c.npy'),'B':np.load('B_all.npy'),'slA':np.load('inv_v21slateA.npy'),'p3':np.load('inv_p3_d16.npy'),'p33b':np.load('inv_p33blend70.npy'),'pbp3':np.load('inv_probe_blend_p3.npy'),'bE0':np.load('inv_best_E0.npy'),'w10':np.load('inv_w10.npy')}
V8=np.load('inv_v8regen.npy');V5=np.load('inv_v5.npy')
allk=list(sets)+list(new); X=np.c_[[sets[k] if k in sets else new[k] for k in allk]].T
idx=np.random.RandomState(0).choice(len(EN),200000,replace=False); RR=np.corrcoef(X[idx].T); m=len(allk); jx={k:i for i,k in enumerate(allk)}
print('rho with EN / R1-ish:'); 
for k in new: print(k,round(RR[0,jx[k]],3),'v8r',round(RR[jx['v8r'],jx[k]],3),'v5',round(RR[jx['v5'],jx[k]],3),'A',round(RR[jx['A'],jx[k]],3))
rs=np.random.RandomState(21); K=3000; sel=rs.choice(len(draws),K,p=W)
Dm=np.zeros((K,m)); Rm=np.zeros((K,m,m))
for t,i in enumerate(sel):
    a,R=draws[i]; d=np.array([dp(x) for x in a]); ao=a[ix['A']]
    ex={'v5c':a[ix['v5']]+rs.uniform(-.003,.003),'B':ao+rs.uniform(-.001,.002),'slA':ao-rs.uniform(0,.0007),'p3':rs.uniform(.778,.782),'p33b':rs.uniform(.8145,.8160),'pbp3':rs.uniform(.805,.815),'bE0':a[ix['E0']],'w10':a[0]}
    Dm[t]=np.r_[d,[dp(ex[k]) for k in new]]
    Rj=RR.copy(); Rj[:n,:n]=R
    for i2 in range(n,m):
        for j2 in range(i2):
            v=min(RR[i2,j2]+rs.normal(0,.0008 if RR[i2,j2]>.99 else .003),.9999); Rj[i2,j2]=Rj[j2,i2]=v
    for i2 in range(n,m):
        for j2 in range(n):
            v=min(RR[i2,j2]+rs.normal(0,.0008 if RR[i2,j2]>.99 else .003),.9999); Rj[i2,j2]=Rj[j2,i2]=v
    Rm[t]=Rj
def A_(w,d=Dm,R=Rm): return auc((d@w)/np.sqrt(np.einsum('i,nij,j->n',w,R,w)))
wR=np.zeros(m); wR[0]=1; wR[1]=-.136; wR[ix['A']]=.049; wR[ix['D']]=.049; wR[ix['v8r']]=-.03; wR[ix['v5']]=-.015
base=A_(wR)
from scipy.optimize import minimize
for k in new:
    if k in('bE0','w10'): continue
    f=lambda v:-(A_(wR+v[0]*np.eye(m)[jx[k]])-base).mean()*1e4
    r=minimize(f,[0.],method='Nelder-Mead'); wo=r.x[0]
    for tag,ww in (('opt',wo),('half',wo/2),('-.02',-.02),('+.02',.02)):
        g=A_(wR+ww*np.eye(m)[jx[k]])-base; print('%-5s %-5s w=%+.3f mean %+.6f 5/95 %+.6f %+.6f P>0 %.2f'%(k,tag,ww,g.mean(),*np.percentile(g,[5,95]),(g>0).mean()))
