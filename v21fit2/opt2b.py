import numpy as np,sys
from scipy.optimize import minimize
sys.path.insert(0,'/workspace/huawei/v21fit')
from binormal import dp,auc
z=np.load(__import__('os').environ.get('POST','post2.npz')); a=z['a']; RR=z['RR']; N=len(a); n=10; m=12; i7,i8=10,11
ks=['EN','E0','A','D','P','v5','v8r','S','slB','listA']; ix={k:i for i,k in enumerate(ks)}
dd=dp(a); aEN=a[:,0]
pS1=np.load('/workspace/huawei/v21fit/pS1.npy')
def mk(p):
    c,g,q,v8,v5=p; w=np.zeros(m); w[0]=1; w[1]=-c; w[ix['A']]=g*(1-q); w[ix['D']]=g*q; w[ix['v8r']]=v8; w[ix['v5']]=v5; return w
def AUCW(W): return auc(np.einsum('ni,ni->n',W,dd)/np.sqrt(np.einsum('ni,nij,nj->n',W,RR,W)))
def Wof(p,w7,w8):
    w=mk(p); s=np.sqrt(np.einsum('i,nij,j->n',w,RR,w)); W=np.repeat(w[None],N,0)/s[:,None]; W[:,i7]=w7; W[:,i8]=w8; return W
gG3=AUCW(Wof(pS1,0,.05)); gG2=AUCW(Wof(pS1,.05,0)); gS1=AUCW(Wof(pS1,0,0))
def gain(p,w7,w8,ref=gG3): return AUCW(Wof(p,w7,w8))-ref
def stats(g): 
    return 'mean %+.6f 5/95 %+.6f %+.6f P>0 %.2f P>5e-5 %.2f'%(g.mean(),*np.percentile(g,[5,95]),(g>0).mean(),(g>5e-5).mean())
if __name__=='__main__':
    print('G3 vs G2',stats(gG3-gG2),'| G3 vs S1',stats(gG3-gS1),'| G2 vs S1',stats(gG2-gS1))
    print('--- graph weight scan on S1 (w7=0):')
    for w in (0,.02,.035,.05,.07,.1): print(' w8=%.3f'%w,stats(gain(pS1,0,w)))
    print(' G7+G8 mix: w7=.025,w8=.025',stats(gain(pS1,.025,.025)),'\n w7=.05,w8=.05',stats(gain(pS1,.05,.05)))
    # full optimisation
    f=lambda x:-gain(x[:5],0,x[5]).mean()*1e5
    x0=np.r_[pS1,.05]
    r=minimize(f,x0,method='Nelder-Mead',options=dict(maxiter=600,xatol=1e-3,fatol=1e-4,initial_simplex=x0+np.vstack([np.zeros(6),np.diag([.02,.02,.1,.02,.01,.02])])))
    xo=r.x; print('opt (c,g,q,v8,v5,w8)',xo.round(3),'->',stats(gain(xo[:5],0,xo[5])),flush=True)
    np.save('xopt.npy',xo)
    for sh in (.25,.5,.75):
        x=x0+sh*(xo-x0); print('shrink %.2f'%sh,x.round(3),stats(gain(x[:5],0,x[5])))
    # holding DCN/A split and total g as S1: optimize c,v8,v5,w8
    def f2(y): 
        c,v8,v5,w8=y; p=pS1.copy(); p[0]=c;p[3]=v8;p[4]=v5; return -gain(p,0,w8).mean()*1e5
    y0=np.array([pS1[0],pS1[3],pS1[4],.05]); r2=minimize(f2,y0,method='Nelder-Mead',options=dict(maxiter=400,initial_simplex=y0+np.vstack([np.zeros(4),np.diag([.02,.02,.01,.02])])))
    p=pS1.copy(); p[0],p[3],p[4]=r2.x[:3]; print('opt with g,q fixed (c,v8,v5,w8)',r2.x.round(3),stats(gain(p,0,r2.x[3])))
    # 1-d scans from S1 coefficients of E0, v8, v5 at w8=.05
    for nm,j in (('E0 c',0),('v8',3),('v5',4)):
        for d in (-.04,-.02,.02,.04):
            p=pS1.copy(); p[j]+=d; print(' %s %+.2f'%(nm,d),stats(gain(p,0,.05)))
