import numpy as np, pickle
from addon import n,ix,wQ,R0
from binormal import dp,auc
from scipy.optimize import minimize
draws,W,bm,bv=pickle.load(open('post.pkl','rb'))
rs=np.random.RandomState(3)
def prep(k):
    sel=rs.choice(len(draws),k,p=W); d=np.array([[dp(x) for x in draws[i][0]] for i in sel]); R=np.array([draws[i][1] for i in sel]); return d,R
dO,RO=prep(2000); dE,RE=prep(6000)
def A_(w,d,R): return auc((d@w)/np.sqrt(np.einsum('i,nij,j->n',w,R,w)))
wR=wQ.copy(); wR[ix['v8r']]=-.03; wR[ix['v5']]=-.015
def mk(p):
    c,g,q,v8,v5=p; w=np.zeros(n); w[0]=1; w[1]=-c; w[ix['A']]=g*(1-q); w[ix['D']]=g*q; w[ix['v8r']]=v8; w[ix['v5']]=v5; return w
pR=np.array([.136,.098,.5,-.03,-.015]); assert np.allclose(mk(pR),wR)
sc=np.array([.1,.05,.5,.05,.05])
base=A_(wR,dO,RO)
def obj(p,lam): 
    return -(A_(mk(p),dO,RO)-base).mean()*1e4 + lam*np.sum(((p-pR)/sc)**2)*1e-2
def rep(p,lab):
    g=A_(mk(p),dE,RE)-A_(wR,dE,RE); print('%-26s p=%s gain vs R1 mean %+.6f 5/95 %+.6f %+.6f P>0 %.2f'%(lab,np.round(p,3),g.mean(),*np.percentile(g,[5,95]),(g>0).mean())); return g
out={}
for lam in (0,.3,1,3):
    r=minimize(obj,pR,args=(lam,),method='Nelder-Mead',options={'maxiter':2500,'xatol':1e-4,'fatol':1e-7}); p=r.x.copy(); p[2]=np.clip(p[2],0,1); out[lam]=p
    rep(p,f'ridge {lam}')
for t in (.5,): rep(pR+t*(out[0]-pR),'50% of unshrunk opt')
print('--- one-at-a-time vs R1')
for nm,i,vals in (('q (DCN share)',2,(.75,1)),('c E0 coef',0,(.1,.18)),('g total ours',1,(.07,.13)),('v8r',3,(-.06,0)),('v5',4,(-.03,0))):
    for v in vals:
        p=pR.copy(); p[i]=v; rep(p,f'{nm}={v}')
pickle.dump(out,open('reopt_out.pkl','wb'))
