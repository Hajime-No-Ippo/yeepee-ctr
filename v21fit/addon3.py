import numpy as np
exec(open('addon2.py').read().split("print('--- single add-ons")[0])
from scipy.optimize import minimize
free=[ix['v8r'],ix['v5']]
for lam in (0,.5,2):
    def f(v):
        w=wQ.copy(); w[free]+=v; return -comb(w,a0,R0)+lam*1e-1*np.sum(v**2)
    r=minimize(f,np.zeros(2),method='Nelder-Mead'); w=wQ.copy(); w[free]+=r.x
    print('ridge',lam,'v8r,v5 =',r.x.round(3),'mean-input gain',round(comb(w,a0,R0)-comb(wQ,a0,R0),6)); mc(w,'  MC')
for a,b in ((-.04,-.02),(-.03,-.03),(-.03,-.015),(-.05,0),(0,-.03)):
    w=wQ.copy(); w[ix['v8r']]=a; w[ix['v5']]=b; mc(w,f'v8r {a} v5 {b}')
# stress: v5 AUC much higher / v8r lower than assumed
rs2=np.random.RandomState(9)
for lab,(lo5,hi5,lo8,hi8) in {'v5 .795-.805 (optimistic)':(.795,.805,.813,.8155),'v8r .808-.812 (pessimistic)':(.784,.794,.808,.812),'both bad':(.795,.805,.808,.812)}.items():
    dd=[]
    for a,R in draws:
        a=a.copy(); a[ix['v5']]=rs2.uniform(lo5,hi5); a[ix['v8r']]=rs2.uniform(lo8,hi8); dd.append((a,R))
    w=wQ.copy(); w[ix['v8r']]=-.03; w[ix['v5']]=-.015
    g=np.array([comb(w,a,R)-comb(wQ,a,R) for a,R in dd])+errs
    print('stress',lab,'(v8r -.03,v5 -.015): mean %+.6f 5/95%% %+.6f %+.6f P>0 %.2f'%(g.mean(),*np.percentile(g,[5,95]),(g>0).mean()))
