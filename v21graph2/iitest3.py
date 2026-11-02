import sys,numpy as np; sys.dont_write_bytecode=True
from ana import *; import ii
for c,tag,step in ((25,'u1',350),(23,'u1b',800)):
    nm={23:'val23',25:'val25'}[c]; k=KOF[c]; A=avg(tag,c,SEEDORD[:7],step); G=sum3(A); line(c,'7seed sum3',G)
    g=np.load(O+f'g_{k}.npz'); T=np.load(O+f'tab_{nm}.npz'); lp=np.log1p(np.bincount(g['ci'],minlength=ii.NI).astype(float))[T['item']]
    for W in (604800,99999999):
        F,_=ii.feats(nm,k,W); x=zs(np.log1p(F['sum']*100))
        for b in (.25,.5,1.0): line(c,f'sum3+{b}*ii(W{W//86400})',G+b*3*x/3*1.0)   # G has 3 z-terms: ii weight b relative to each
        line(c,f'ii alone W{W//86400} +.25pop',x+.25*zs(lp))
