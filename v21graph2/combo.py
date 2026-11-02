import sys,itertools,numpy as np; sys.dont_write_bytecode=True
from ana import *; import ana; _B=ana._B
n=int(sys.argv[1]) if len(sys.argv)>1 else 7
S={25:('u1',350),23:('u1b',800)}; Z={}
for c,(tag,st) in S.items():
    sd=[s for s in SEEDORD[:n]] if c==25 else [s for s in SEEDORD if s!=10][:n]
    A=avg(tag,c,sd,st); Z[c]=np.stack([zs(A[:,j]) for j in range(4)],1); best(c,Z[c][:,0])  # init cache
def g(c,G,w=.125):
    y,B,M=_B[c]; Gn=nr(G); return np.mean([(AUC(y,(1-w)*b+w*Gn)-AUC(y,b)) for b in B.values()])*1e5
res=[]
for wts in itertools.product([0,.5,1,2],repeat=4):
    if sum(wts)==0 or max(wts)<1: continue
    v=np.array(wts,float); r=[g(c,Z[c]@v) for c in (25,23)]; res.append((min(r),r,wts))
res.sort(reverse=True)
for m,r,w in res[:12]: print('wts s_uv,s_h,cos_mean,cos_max',w,'C25 %.1f C23 %.1f'%tuple(r))
print('baseline (1,1,1,0):',[round(g(c,Z[c]@np.array([1,1,1,0.])),1) for c in (25,23)])
for w in ((1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1),(1,1,0,0),(1,1,1,1)): print(w,[round(g(c,Z[c]@np.array(w,float)),1) for c in (25,23)])
