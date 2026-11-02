import sys,os,numpy as np; sys.dont_write_bytecode=True
from evalg import *; from gfeat import *
KOF={23:3,25:5,26:6,27:7}; TAB={23:'val23',25:'val25',26:'val26',27:'eval'}
NAMES=('s_uv','s_h','cos_mean','cos_max')
SEEDORD=[0,1,2,3,4,5,9,10,6,7,8,11]
def zs(x): return (x-x.mean())/x.std()
def feat(tag,c,seed,step):
    """cached raw features (n,4) float32 for emb_{tag}_k{k}_s{seed}.npz step"""
    cf=O+f'fc_{tag}_c{c}_s{seed}_{step}.npy'
    if os.path.exists(cf): return np.load(cf)
    e=np.load(O+f'emb_{tag}_k{KOF[c]}_s{seed}.npz'); F=feats(TAB[c],e[f'U{step}'],e[f'V{step}'])
    a=np.stack([F[k] for k in NAMES],1).astype(np.float32); np.save(cf,a); return a
def avg(tag,c,seeds,step):
    acc=0
    for s in seeds: acc=acc+feat(tag,c,s,step).astype(np.float64)/len(seeds)
    return acc
def sum3(A): return zs(A[:,0])+zs(A[:,1])+zs(A[:,2])
WS=(.05,.075,.1,.125,.15,.2)
_B={}
def best(c,G,ws=WS):
    """returns auc, rho_M, and per base (best gain e-5, best w)"""
    if c not in _B:
        y,z,M,b=ref(c); y_,B=bases(c); _B[c]=(y,B,nr(M))
    y,B,M=_B[c]; Gn=nr(G); out={}
    for k,b in B.items():
        a0=AUC(y,b); gs=[AUC(y,(1-w)*b+w*Gn)-a0 for w in ws]; i=int(np.argmax(gs)); out[k]=(gs[i]*1e5,ws[i],[round(g*1e5) for g in gs])
    return AUC(y,G),np.corrcoef(Gn,M)[0,1],out
def line(c,name,G):
    a,r,o=best(c,G); print(f'C{c} {name:28s} auc {a:.5f} rhoM {r:.3f} '+' | '.join(f'{k}: best {v[0]:+.1f}@{v[1]} {v[2]}' for k,v in o.items()),flush=True)
    return a,r,o
if __name__=='__main__':
    if sys.argv[1]=='a':
        for c in (25,23):
            tag='u1' if c==25 else 'u1b'; step=350 if c==25 else 800
            for n in (3,5,7,9,10,12):
                sd=SEEDORD[:n]
                try: A=avg(tag,c,sd,step)
                except FileNotFoundError as ex: print('missing',ex); continue
                line(c,f'{n} seeds sum3',sum3(A))
