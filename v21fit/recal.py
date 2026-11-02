import numpy as np, pickle
from addon import *
wR=wQ.copy(); wR[ix['v8r']]=-.03; wR[ix['v5']]=-.015
wQ2=np.zeros(n); wQ2[0]=1; wQ2[1]=-.136; wQ2[ix['D']]=.098
obs=np.array([.829058-.828856,.829016-.828856])
rs=np.random.RandomState(11); N=30000; S=[]
a0=np.array([.828856,.824558,.8197,.8185,.805,.789,.8144,.8197,.8192,.8192])
print('prior-mean preds  Q1 %.6f Q2 %.6f R1 %.6f (gain vs EN)'%tuple(comb(w,a0,R0)-.828856 for w in (wQ,wQ2,wR)))
draws=[]
while len(draws)<N:
    a,R=draw(rs)
    if np.linalg.eigvalsh(R).min()>1e-6: draws.append((a,R))
pred=np.array([[comb(w,a,R)-a[0] for w in (wQ2,wR,wQ)] for a,R in draws])  # Q2,R1,Q1
sig,tau=.00006,.00005
C=sig**2*np.eye(2)+tau**2*np.ones((2,2)); Ci=np.linalg.inv(C)
res=obs[None,:]-pred[:,:2]; ll=-.5*np.einsum('ni,ij,nj->n',res,Ci,res); W=np.exp(ll-ll.max()); W/=W.sum()
print('ESS',1/np.sum(W**2))
# conditional bias b | draw: prior N(0,tau^2), obs = pred + b + e
bm=(res.sum(1)*tau**2/(sig**2+2*tau**2)); bv=tau**2*sig**2/(sig**2+2*tau**2)  # posterior mean/var (approx for 2 obs)
def wm(x): return (W*x).sum()
def wq(x,q): 
    o=np.argsort(x); c=np.cumsum(W[o]); return x[o][np.searchsorted(c,q)]
for nm,f in (('aD',lambda a:a[ix['D']]),('aA/ours',lambda a:a[ix['A']]),('aV5',lambda a:a[ix['v5']]),('aV8r',lambda a:a[ix['v8r']]),('aP',lambda a:a[ix['P']])):
    pr=np.array([f(a) for a,_ in draws]); print('%-8s prior mean %.5f  posterior mean %.5f  5/50/95 %.5f %.5f %.5f'%(nm,pr.mean(),wm(pr),wq(pr,.05),wq(pr,.5),wq(pr,.95)))
print('bias b: posterior mean %.6f sd %.6f'%(wm(bm),np.sqrt(wm(bv+bm**2)-wm(bm)**2)))
print('posterior predicted gain vs EN (excl bias): Q2 %.6f R1 %.6f Q1 %.6f'%(wm(pred[:,0]),wm(pred[:,1]),wm(pred[:,2])))
print('Q1 incl bias: mean %.6f; 5/95 approx'%(wm(pred[:,2]+bm)), wq(pred[:,2]+bm,.05),wq(pred[:,2]+bm,.95))
print('Q1 public AUC predicted: %.6f'%(.828856+wm(pred[:,2]+bm)))
pickle.dump((draws,W,bm,bv),open('post.pkl','wb'))
