import sys,numpy as np; sys.dont_write_bytecode=True
from ana import *; import ana
# usage: combo3.py seeds   (L8 u12, step 650; k3 uses u10b s0 only until u12 k3 seeds exist)
seeds=[int(s) for s in sys.argv[1].split(',')]
Z={}
for c in (25,23):
    try:
        A=avg('u12',c,seeds,650); Z[c]=np.stack([zs(A[:,j]) for j in range(4)],1); 
    except Exception as e: print('skip',c,e)
def g(c,G,w):
    y,B,M=ana._B[c]; Gn=nr(G); return [(AUC(y,(1-w)*b+w*Gn)-AUC(y,b))*1e5 for b in B.values()]
for c in Z: best(c,Z[c]@np.array((1,1,1,0.)))
for wts in ((1,1,1,0),(2,0,1,0),(1,0,1,0),(2,.5,1,0),(1,1,0,0)):
    v=np.array(wts,float)
    for w in (.1,.15,.2):
        print(wts,'w',w,' | '.join('C%d %s'%(c,' '.join('%.1f'%x for x in g(c,Z[c]@v,w))) for c in Z),flush=True)
