import sys,numpy as np; sys.dont_write_bytecode=True
from ana import *; import ii
for c in (25,23):
    nm={23:'val23',25:'val25'}[c]; k=KOF[c]; T=np.load(O+f'tab_{nm}.npz'); it=T['item']
    g=np.load(O+f'g_{k}.npz'); pop=np.bincount(g['ci'],minlength=ii.NI).astype(float); lp=np.log1p(pop)[it]
    print(f'C{c} pop auc {AUC(T["y"],lp):.5f}')
    for W in (604800,99999999):
        F,yy=ii.feats(nm,k,W); s=F['sum']; 
        for b in (0,.25,.5,1,2):
            x=zs(np.log1p(s*100))+b*zs(lp) if True else 0; print(f'C{c} W={W} sum(log)+{b}*pop auc {AUC(yy,x):.5f}',flush=True)
