import numpy as np,sys
from lib import *
from binormal import dp,auc
from scipy.optimize import minimize
A=np.load('A_all.npy'); D=np.load('D_dcn.npy')
sets={'EN':EN,'E0':E0,'A':A,'D':D,'P':P,'v5':np.load('inv_v5.npy'),'v8r':np.load('inv_v8regen.npy'),'S':np.load('S_ebdistill.npy'),'slB':np.load('inv_v21slateB.npy'),'listA':np.load('inv_v21listA.npy')}
ks=list(sets); X=np.c_[[sets[k] for k in ks]].T; n=len(ks); ix={k:i for i,k in enumerate(ks)}
idx=np.random.RandomState(0).choice(len(EN),200000,replace=False); R0=np.corrcoef(X[idx].T)
np.save('R0.npy',R0); print(ks); print(R0.round(3))
wQ=np.zeros(n); wQ[ix['EN']]=1; wQ[ix['E0']]=-.136; wQ[ix['A']]=.049; wQ[ix['D']]=.049
def draw(rs):
    a=np.zeros(n); ao=rs.uniform(.8185,.8205)
    a[ix['EN']]=.828856; a[ix['E0']]=.824558; a[ix['A']]=ao; a[ix['D']]=ao-rs.uniform(0,.0025)
    a[ix['P']]=rs.uniform(.800,.810); a[ix['v5']]=rs.uniform(.784,.794); a[ix['v8r']]=rs.uniform(.813,.8155)
    a[ix['S']]=ao+rs.uniform(-.001,.002); a[ix['slB']]=ao-rs.uniform(0,.0007); a[ix['listA']]=ao-rs.uniform(0,.0007)
    R=R0.copy()
    for i in range(n):
        for j in range(i+1,n):
            sd=.0008 if R0[i,j]>.99 else .003
            v=min(R0[i,j]+rs.normal(0,sd),.9999); R[i,j]=R[j,i]=v
    return a,R
def comb(w,a,R): d=np.array([dp(x) for x in a]); return auc(w@d/np.sqrt(w@R@w))
if __name__=='__main__':
    rs=np.random.RandomState(5); N=1500
    draws=[]
    while len(draws)<N:
        a,R=draw(rs)
        if np.linalg.eigvalsh(R).min()>1e-6: draws.append((a,R))
    a0=np.array([.828856,.824558,.8197,.8185,.805,.805,.814,.8197,.8192,.8192]); 
    print('mean-input Q1 pred',round(comb(wQ,a0,R0),6))
    errs=rs.normal(0,.00003,N)
    print('--- add-on to Q1: opt weight (mean inputs), then paired MC gain over Q1 at that weight and at half weight')
    for k in ks[1:]:
        if k in('EN',): continue
        f=lambda v:-comb(wQ+v[0]*np.eye(n)[ix[k]],a0,R0); r=minimize(f,[0],method='Nelder-Mead'); wo=r.x[0]
        for tag,ww in (('opt',wo),('half',wo/2),('fixed+.03',.03),('fixed-.03',-.03)):
            if tag.startswith('fixed') and tag!='fixed+.03' and tag!='fixed-.03': continue
            g=np.array([comb(wQ+ww*np.eye(n)[ix[k]],a,R)-comb(wQ,a,R) for a,R in draws])+errs
            print('%-6s %-10s w=%+.3f  mean %+.6f  5/95%% %+.6f %+.6f  P>0 %.2f'%(k,tag,ww,g.mean(),*np.percentile(g,[5,95]),(g>0).mean()))
