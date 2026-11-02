exec(open('reopt3.py').read().split("# stress on v5")[0].replace("for t in (.25,.33,.5,.67): rep(pR+t*(p0-pR),f'step {t}')",""))
def obj2(x): p=pR.copy(); p[[0,1,2,3]]=x; return -(A_(mk(p),dO,RO)-base).mean()*1e4
r=minimize(obj2,pR[:4],method='Nelder-Mead',options={'maxiter':2500}); q=pR.copy(); q[:4]=r.x; q[2]=min(q[2],1); print('v5-fixed opt',q.round(3))
dH=dE.copy(); dH[:,ix['v5']]=dp(np.random.RandomState(1).uniform(.795,.805,len(dH)))
for t in (.5,1):
    p=pR+t*(q-pR); rep(p,f'v5fixed step {t}'); rep(p,f'  stress v5 hi',dH,RE)
p=pR+.33*(p0-pR); rep(p,'S1 (1/3 full)'); 
np.save('pS1.npy',p); np.save('pS2.npy',pR+.5*(q-pR))
