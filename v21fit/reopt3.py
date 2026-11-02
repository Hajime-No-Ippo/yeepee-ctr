exec(open('reopt2.py').read().split("out={}")[0])
import pickle
out=pickle.load(open('reopt_out.pkl','rb')); p0=out[1]
def rep(p,lab,d=dE,R=RE):
    g=A_(mk(p),d,R)-A_(wR,d,R); print('%-14s p=%s mean %+.6f 5/95 %+.6f %+.6f P>0 %.2f'%(lab,np.round(p,3),g.mean(),*np.percentile(g,[5,95]),(g>0).mean())); return g
for t in (.25,.33,.5,.67): rep(pR+t*(p0-pR),f'step {t}')
# stress on v5 high (>=.795) using posterior draws but overriding v5 AUC
dS=dE.copy(); dS[:,ix['v5']]=dp(np.random.RandomState(1).uniform(.795,.805,len(dS)))
for t in (.25,.5): rep(pR+t*(p0-pR),f'stress v5 hi {t}',dS,RE)
dS=dE.copy(); dS[:,ix['v8r']]=dp(np.random.RandomState(1).uniform(.808,.812,len(dS)))
for t in (.25,.5): rep(pR+t*(p0-pR),f'stress v8r lo {t}',dS,RE)
