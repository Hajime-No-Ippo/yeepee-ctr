import sys,numpy as np; sys.dont_write_bytecode=True
from opt2 import *
import itertools
def nrm(w): return w/np.sqrt(np.einsum('ni,nij,nj->n',w,RR,w))[:,None]
def B(p): return nrm(np.repeat(mk(p)[None],N,0))
def with_g(W,w7,w8): W=nrm(W).copy(); W[:,i7]=w7; W[:,i8]=w8; return W
def g_(W): return AUCW(W)-gG3
def rep(lab,W):
    g=g_(W); print('%-44s mean %+.6f 5/95 %+.6f %+.6f P>0 %.2f P>5e-5 %.2f'%(lab,g.mean(),*np.percentile(g,[5,95]),(g>0).mean(),(g>5e-5).mean()),flush=True); return g.mean(),lab,g
res=[]
pQ2=np.array([.136,.098,1,0,0]); pR1=np.array([.136,.098,.5,-.03,-.015])
print('=== DIR1: S1 + w7*G7 + w8*G8 (unit-sd graph weights, G3=(0,.05))')
d1=[]
for tot in (.04,.05,.06,.07,.08):
    for s7 in (0,.25,.5,.75,1):
        d1.append(rep('tot %.2f share7 %.2f (w7 %.3f w8 %.3f)'%(tot,s7,tot*s7,tot*(1-s7)),with_g(B(pS1),tot*s7,tot*(1-s7))))
res+=[('d1',)+x for x in d1]
print('=== DIR2: lower ours total g, raise graph (E0/v8/v5 at S1; q kept .665)')
d2=[]
for g in (.123,.10,.08,.06,.04):
    for w8 in (.05,.07,.09,.12,.15):
        p=pS1.copy(); p[1]=g; d2.append(rep('g %.3f w8 %.2f'%(g,w8),with_g(B(p),0,w8)))
print(' with 7s/G8 half-half graph:')
for g in (.10,.08,.06):
    for tot in (.08,.12):
        p=pS1.copy(); p[1]=g; d2.append(rep('g %.3f tot %.2f half/half'%(g,tot),with_g(B(p),tot/2,tot/2)))
print(' lightly refit (c,v8,v5,w8) at fixed g:')
from scipy.optimize import minimize
for g in (.10,.08,.06):
    def f(y):
        p=pS1.copy(); p[1]=g; p[0],p[3],p[4]=y[:3]; return -g_(with_g(B(p),0,y[3])).mean()*1e5
    y0=np.r_[pS1[0],pS1[3],pS1[4],.08]; r=minimize(f,y0,method='Nelder-Mead',options=dict(maxiter=250,initial_simplex=y0+np.vstack([np.zeros(4),np.diag([.02,.02,.01,.02])])))
    p=pS1.copy(); p[1]=g; p[0],p[3],p[4]=r.x[:3]; d2.append(rep('refit g %.3f c %.3f v8 %.3f v5 %.3f w8 %.3f'%(g,*r.x),with_g(B(p),0,r.x[3])))
res+=[('d2',)+x for x in d2]
print('=== DIR3: rank-avg G3 with Q2 / R1, weight m, + extra graph on top')
G3W=with_g(B(pS1),0,.05); d3=[]
for nmq,pp in (('Q2',pQ2),('R1',pR1)):
    for mm in (.1,.2,.3,.5):
        Wm=(1-mm)*G3W+mm*B(pp)
        d3.append(rep('G3 + %s m=%.1f'%(nmq,mm),nrm(Wm)))
        for ex in (.03,.06):
            d3.append(rep('  +%s m=%.1f extra G8 %.2f'%(nmq,mm,ex),with_g(nrm(Wm),0,ex)))
    # base without graph mixed then graph
    for mm in (.2,.4):
        Wm=(1-mm)*B(pS1)+mm*B(pp)
        d3.append(rep('S1+%s m=%.1f then G8 .05'%(nmq,mm),with_g(nrm(Wm),0,.05)))
res+=[('d3',)+x for x in d3]
for d in ('d1','d2','d3'):
    r=[x for x in res if x[0]==d]; b=max(r,key=lambda x:x[1]); g=b[3]
    print('BEST',d,b[2],'mean %+.6f 5/95 %+.6f %+.6f P>0 %.2f P>5e-5 %.2f'%(g.mean(),*np.percentile(g,[5,95]),(g>0).mean(),(g>5e-5).mean()))
