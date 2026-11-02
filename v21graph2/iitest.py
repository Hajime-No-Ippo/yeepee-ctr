import sys,numpy as np; sys.dont_write_bytecode=True
from ana import *; import ii
c=int(sys.argv[1]); nm={23:'val23',25:'val25'}[c]; k=KOF[c]
y=_=None
for W in [int(x) for x in sys.argv[2:]]:
    for alpha in (.5,):
        F,yy=ii.feats(nm,k,W,alpha=alpha); 
        for key,s in F.items():
            print(f'C{c} W={W} a={alpha} {key:6s} auc {AUC(yy,s):.5f}',flush=True)
