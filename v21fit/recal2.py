import numpy as np, pickle
from addon import *
from scipy.stats import norm
pR=np.array([.136,.098,.5,-.03,-.015]); pS1=np.load('pS1.npy'); pS2=np.load('pS2.npy'); p0=pickle.load(open('reopt_out.pkl','rb'))[1]
def mk(p):
    c,g,q,v8,v5=p; w=np.zeros(n); w[0]=1; w[1]=-c; w[ix['A']]=g*(1-q); w[ix['D']]=g*q; w[ix['v8r']]=v8; w[ix['v5']]=v5; return w
W_={'Q2':mk([.136,.098,1,0,0]),'R1':mk(pR),'S1':mk(pS1),'Q1':mk([.136,.098,.5,0,0]),'S2':mk(pS2),'T1':mk(pR+.667*(p0-pR)),'T2':mk(p0),'T3':mk(pR+.75*(pR*0+(pR+2*(pS2-pR))-pR))}
names=list(W_); obsn=['Q2','R1','S1']; obs=np.array([.000160,.000202,.000245])
rs=np.random.RandomState(31); N=30000; draws=[]
while len(draws)<N:
    a,R=draw(rs)
    if np.linalg.eigvalsh(R).min()>1e-6: draws.append((a,R))
pred={k:np.array([comb(W_[k],a,R)-a[0] for a,R in draws]) for k in names}
P=np.c_[[pred[k] for k in obsn]].T
for sig in (.00004,.00006):
  tau=.00005
  C=sig**2*np.eye(3)+tau**2*np.ones((3,3)); Ci=np.linalg.inv(C)
  res=obs[None,:]-P; ll=-.5*np.einsum('ni,ij,nj->n',res,Ci,res); Wt=np.exp(ll-ll.max()); Wt/=Wt.sum()
  print('=== sig',sig,'ESS %.0f'%(1/np.sum(Wt**2)))
  # bias posterior (mean of b | draw, 3 obs)
  bm=res.sum(1)*tau**2/(sig**2+3*tau**2)
  wm=lambda x:(Wt*x).sum()
  def wq(x,q): o=np.argsort(x); c=np.cumsum(Wt[o]); return x[o][min(np.searchsorted(c,q),len(x)-1)]
  for nm,f in (('aD',lambda a:a[ix['D']]),('ours',lambda a:a[ix['A']]),('aV5',lambda a:a[ix['v5']]),('aV8r',lambda a:a[ix['v8r']]),('aP',lambda a:a[ix['P']])):
      pr=np.array([f(a) for a,_ in draws]); print('%-5s prior %.5f post %.5f  5/95 %.5f %.5f'%(nm,pr.mean(),wm(pr),wq(pr,.05),wq(pr,.95)))
  print('bias mean %.6f'%wm(bm),'| fit pred vs obs (gain vs EN):',{k:round(wm(pred[k]+bm),6) for k in obsn})
  print('Q1 pred gain vs EN %.6f (AUC %.6f)'%(wm(pred['Q1']+bm),.828856+wm(pred['Q1']+bm)))
  for k in ('T1','T2','T3','S2'):
      for ref in ('S1','R1'):
          if k==ref: continue
          g=pred[k]-pred[ref]; print('%-3s vs %-2s mean %+.6f 5/95 %+.6f %+.6f P>0 %.2f P>=.00005 %.2f'%(k,ref,wm(g),wq(g,.05),wq(g,.95),wm(g>0),wm(g>=.00005)))
