import numpy as np
from addon import *
rs=np.random.RandomState(7); N=1500; draws=[]
while len(draws)<N:
    a,R=draw(rs)
    if np.linalg.eigvalsh(R).min()>1e-6: draws.append((a,R))
errs=rs.normal(0,.00003,N)
a0=np.array([.828856,.824558,.8197,.8185,.805,.789,.8144,.8197,.8192,.8192])
def mc(w,label):
    g=np.array([comb(w,a,R)-comb(wQ,a,R) for a,R in draws])+errs
    print('%-48s mean %+.6f 5/95%% %+.6f %+.6f P>0 %.2f'%(label,g.mean(),*np.percentile(g,[5,95]),(g>0).mean())); return g
print('--- single add-ons, corrected v5 (.784-.794) / v8regen (.813-.8155)')
for k in ('v5','v8r','P','S'):
    for ww in ((-.03,-.055) if k in('v8r','P','S') else (.03,-.03)):
        mc(wQ+ww*np.eye(n)[ix[k]],f'{k} {ww:+.3f}')
print('--- Q1 internal: grid c(E0 coef) x g(total ours) x q(DCN share); score under mean inputs, then MC of best')
best=None
for c in (0,.07,.136,.2,.27):
    for g in (.06,.098,.14,.2):
        for q in (0,.25,.5,.75,1):
            w=np.zeros(n); w[0]=1; w[1]=-c; w[ix['A']]=g*(1-q); w[ix['D']]=g*q
            s=comb(w,a0,R0)
            if best is None or s>best[0]: best=(s,c,g,q,w)
s,c,g,q,w=best; print('best mean-input pred',round(s,6),'c,g,q',c,g,q,'Q1',round(comb(wQ,a0,R0),6))
mc(w,'grid-best')
ws=wQ+.5*(w-wQ); mc(ws,'grid-best shrunk 50% toward Q1')
for c_ in (.07,.2): 
    w2=wQ.copy(); w2[1]=-c_; mc(w2,f'Q1 with E0 coef {c_}')
for g_ in (.06,.14):
    w2=wQ*1.0; w2[ix['A']]=g_/2; w2[ix['D']]=g_/2; mc(w2,f'Q1 with total ours g={g_}')
for q_ in (0,1):
    w2=wQ*1.0; w2[ix['A']]=.098*(1-q_); w2[ix['D']]=.098*q_; mc(w2,f'Q1 with DCN share q={q_}')
print('--- joint opt (E0,A,D,v8r,P) with ridge toward Q1, mean inputs')
from scipy.optimize import minimize
free=[ix[k] for k in ('E0','A','D','v8r','P')]
for lam in (0,.5,2):
    f=lambda v:(lambda w:-comb(w,a0,R0)+lam*1e-3*np.sum((w-wQ)**2)*10)(np.where(np.isin(np.arange(n),free),0,wQ)+np.bincount(free,weights=v,minlength=n)+0)
    # simpler: parametrize deltas
    def f(v):
        w=wQ.copy(); w[free]+=v; return -comb(w,a0,R0)+lam*1e-3*np.sum(v**2)*100
    r=minimize(f,np.zeros(len(free)),method='Nelder-Mead',options={'maxiter':3000}); w=wQ.copy(); w[free]+=r.x
    print('ridge',lam,'delta',dict(zip(('E0','A','D','v8r','P'),r.x.round(3))),'pred gain',round(comb(w,a0,R0)-comb(wQ,a0,R0),6)); mc(w,f'  joint ridge {lam} MC')
