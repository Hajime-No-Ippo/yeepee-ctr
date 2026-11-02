import sys,numpy as np; sys.dont_write_bytecode=True
from ana import *; import ana
S={25:('u1',350),23:('u1b',800)}
for n in (7,12):
    Z={}
    for c,(tag,st) in S.items():
        A=avg(tag,c,SEEDORD[:n],st); Z[c]=np.stack([zs(A[:,j]) for j in range(4)],1); best(c,Z[c][:,0])
    def g(c,G,w):
        y,B,M=ana._B[c]; Gn=nr(G); return [(AUC(y,(1-w)*b+w*Gn)-AUC(y,b))*1e5 for b in B.values()]
    for wts in ((1,1,1,0),(2,0,1,0),(2,.5,1,0),(2,.5,.5,0),(1,1,0,0)):
        v=np.array(wts,float)
        for w in (.1,.125,.15,.2):
            print(n,wts,'w',w,' | '.join('C%d base %.1f stackD %.1f'%((c,)+tuple(g(c,Z[c]@v,w))) for c in (25,23)),flush=True)
