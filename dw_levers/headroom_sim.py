import numpy as np, json
from common import *
rng=np.random.default_rng(0); out={}
for c in (25,23):
    m=meta(c); y=m['clicked'].to_numpy(); z=z12(c); n=len(y)//10; Z=z.reshape(n,10); yev=y.reshape(n,10).max(1)
    pany=1-np.prod(1-1/(1+np.exp(-Z)),1); e0=np.log(pany/(1-pany)); base_ev=A(yev,e0); base=A(y,z); res=[]
    for g in (0.0,0.1,0.2,0.3,0.5,0.8,1.2):
        s=(yev-.5)*g+rng.normal(size=n); llr=g*s   # Bayes update for N(+-g/2,1) side-information
        # row logit: p_row=q*softmax -> shifting event log-odds by llr approx shifts row logit by llr*(1-q)
        q=pany[:,None]; zz=(Z+llr[:,None]*(1-q)).ravel()
        res.append(dict(g=g,event_auc=round(A(yev,e0+llr),4),global_auc=round(A(y,zz),5),delta=round(A(y,zz)-base,5)))
    out[c]=dict(base_global=base,base_event=base_ev,rows=res); print(c,base_ev,json.dumps(res),flush=True)
json.dump(out,open('headroom_sim.json','w'))
