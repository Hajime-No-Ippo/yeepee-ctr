import numpy as np, pickle, resource
from addon import *
pR=np.array([.136,.098,.5,-.03,-.015]); pS1=np.load('pS1.npy'); pS2=np.load('pS2.npy'); p0=pickle.load(open('reopt_out.pkl','rb'))[1]
def mk(p):
    c,g,q,v8,v5=p; w=np.zeros(n); w[0]=1; w[1]=-c; w[ix['A']]=g*(1-q); w[ix['D']]=g*q; w[ix['v8r']]=v8; w[ix['v5']]=v5; return w
pT1=pR+.667*(p0-pR)
obsn={'Q2':(mk([.136,.098,1,0,0]),.000160),'R1':(mk(pR),.000202),'S1':(mk(pS1),.000245),'T1':(mk(pT1),.000240)}
obs=np.array([v[1] for v in obsn.values()])
rs=np.random.RandomState(41); N=36000; draws=[]
while len(draws)<N:
    a,R=draw(rs)
    if np.linalg.eigvalsh(R).min()>1e-6: draws.append((a,R))
dd=np.array([[dp(x) for x in a] for a,_ in draws]); RR=np.array([R for _,R in draws]); a_EN=np.array([a[0] for a,_ in draws])
def G(w): return auc((dd@w)/np.sqrt(np.einsum('i,nij,j->n',w,RR,w)))-a_EN
P=np.c_[[G(v[0]) for v in obsn.values()]].T
tau=.00005; out={}
for sig in (.00004,.00006):
    C=sig**2*np.eye(4)+tau**2*np.ones((4,4)); Ci=np.linalg.inv(C)
    res=obs[None,:]-P; ll=-.5*np.einsum('ni,ij,nj->n',res,Ci,res); Wt=np.exp(ll-ll.max()); Wt/=Wt.sum()
    ess=1/np.sum(Wt**2); bm=res.sum(1)*tau**2/(sig**2+4*tau**2)
    wm=lambda x:(Wt*x).sum()
    def wq(x,q): o=np.argsort(x); c=np.cumsum(Wt[o]); return x[o][min(np.searchsorted(c,q),len(x)-1)]
    print('=== sig',sig,'ESS %.0f'%ess,'RSS MB',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss//1024)
    for nm,k in (('aD','D'),('ours','A'),('aV5','v5'),('aV8r','v8r')):
        pr=np.array([a[ix[k]] for a,_ in draws]); print('%-5s post %.5f  5/95 %.5f %.5f'%(nm,wm(pr),wq(pr,.05),wq(pr,.95)))
    print('fit gains',{k:round(wm(P[:,i]+bm),6) for i,k in enumerate(obsn)},'obs',obs,'bias %.6f'%wm(bm))
    gS1=G(mk(pS1))
    print('line t (R1=0,S1=.33,T1=.667,full=1) gain vs S1:')
    for t in (0,.15,.25,.33,.45,.55,.667,.8,1.0):
        g=G(mk(pR+t*(p0-pR)))-gS1; print('  t=%.2f mean %+.6f 5/95 %+.6f %+.6f P>0 %.2f'%(t,wm(g),wq(g,.05),wq(g,.95),wm(g>0)))
    # free optimum on posterior (5 params), nelder mead on weighted mean, 6000-sample subset for speed
    sel=np.random.RandomState(1).choice(N,6000,p=Wt); d6=dd[sel]; R6=RR[sel]; e6=a_EN[sel]
    g6=auc((d6@mk(pS1))/np.sqrt(np.einsum('i,nij,j->n',mk(pS1),R6,mk(pS1))))
    f=lambda p:-(auc((d6@mk(p))/np.sqrt(np.einsum('i,nij,j->n',mk(p),R6,mk(p))))-g6).mean()*1e4
    from scipy.optimize import minimize
    r=minimize(f,pS1,method='Nelder-Mead',options={'maxiter':1500,'xatol':1e-4,'fatol':1e-8}); po=r.x; po[2]=min(max(po[2],0),1)
    print('refit optimum p=',np.round(po,3),'pred gain vs S1 (6000 resample) %+.6f'%(-f(po)*1e-4))
    for sh in (1,.5):
        p=pS1+sh*(po-pS1); g=G(mk(p))-gS1; print('  shrink %.1f p=%s gain vs S1 mean %+.6f 5/95 %+.6f %+.6f P>0 %.2f'%(sh,np.round(p,3),wm(g),wq(g,.05),wq(g,.95),wm(g>0)))
    out[sig]=(po,sel)
    if sig==.00006:
        np.savez('post5.npz',d=dd[sel[:3000]],R=RR[sel[:3000]],a_EN=a_EN[sel[:3000]])
