"""ms.py cuts: multi-seed linear graph blend gains. python ms.py 25 23"""
import sys,glob,numpy as np; sys.dont_write_bytecode=True
from evalg import *; from gfeat import *
import evalg
def report(c,name,G,ws=(.05,.1,.15,.2,.3)):
    y,r=gains(c,G,ws); rho=np.corrcoef(nr(G),nr(ref(c)[2]))[0,1]
    print(f'C{c} {name}: auc {AUC(y,G):.5f} rho_M {rho:.3f} '+' | '.join(f'{k}: '+' '.join(f'w{w}:{g*1e5:+.0f}' for w,g in v[1].items()) for k,v in r.items()),flush=True)
KOF={23:3,25:5,26:6}; TAGC={23:'u1b',25:'u1',26:'u1'}; STEP={23:'800',25:'350',26:'350'}
NAMES=('s_uv','s_h','cos_mean')
def zs(x): return (x-x.mean())/x.std()
def multiseed(c):
    acc=None;n=0
    for f in sorted(glob.glob(O+f'emb_{TAGC[c]}_k{KOF[c]}_s*.npz')):
        e=np.load(f)
        if f'U{STEP[c]}' not in e.files: continue
        F=feats(CUTS[c],e[f'U{STEP[c]}'],e[f'V{STEP[c]}']); n+=1
        acc={k:F[k].astype(np.float64) for k in F} if acc is None else {k:(acc[k]+F[k] if k in NAMES else acc[k]) for k in acc}
    for k in NAMES: acc[k]/=n
    return acc,n
for c in [int(x) for x in sys.argv[1:]]:
    F,n=multiseed(c); y,z,M,b=ref(c)
    G=zs(demean(F['s_uv']))+zs(demean(F['s_h']))+zs(demean(F['cos_mean']))
    np.save(O+f'graph_simple_c{c}.npy',G.astype(np.float32))
    print(f'seeds {n}',end=' '); report(c,'dm sum3',G)
    report(c,'dm s_uv+s_h',zs(demean(F['s_uv']))+zs(demean(F['s_h'])))
    report(c,'dm cos_mean',demean(F['cos_mean']))
