"""final.py cuts...: fixed 7-seed raw graph score. cuts in 23 25 26 27(eval). saves out/graph_c{c}.npy (float32); eval -> out/graph_eval_raw.npy (tab_eval order = v21seq seqfeat order)"""
import sys,glob,numpy as np; sys.dont_write_bytecode=True
from evalg import *; from gfeat import *
SEEDS=[0,1,2,3,4,5,9]
KOF={23:3,25:5,26:6,27:7}; TAGC={23:'u1b',25:'u1',26:'u1',27:'u1'}; STEP={23:'800',25:'350',26:'350',27:'350'}; TAB={23:'val23',25:'val25',26:'val26',27:'eval'}
NAMES=('s_uv','s_h','cos_mean')
def zs(x): return (x-x.mean())/x.std()
def score(c,seeds=SEEDS):
    acc={k:0 for k in NAMES}
    for s in seeds:
        e=np.load(O+f'emb_{TAGC[c]}_k{KOF[c]}_s{s}.npz'); F=feats(TAB[c],e[f'U{STEP[c]}'],e[f'V{STEP[c]}'])
        for k in NAMES: acc[k]=acc[k]+F[k].astype(np.float64)/len(seeds)
    return acc
def report(c,name,G,ws=(.05,.1,.125,.15,.2)):
    y,r=gains(c,G,ws); rho=np.corrcoef(nr(G),nr(ref(c)[2]))[0,1]
    print(f'C{c} {name}: auc {AUC(y,G):.5f} rho_M {rho:.3f} '+' | '.join(f'{k}: '+' '.join(f'w{w}:{g*1e5:+.0f}' for w,g in v[1].items()) for k,v in r.items()),flush=True)
if __name__=='__main__':
    for c in [int(x) for x in sys.argv[1:]]:
        A=score(c); G=zs(A['s_uv'])+zs(A['s_h'])+zs(A['cos_mean'])
        np.save(O+f'graph_c{c}.npy',G.astype(np.float32))
        if c in (23,25):
            report(c,'raw sum3',G); report(c,'raw s_uv+s_h',zs(A['s_uv'])+zs(A['s_h'])); report(c,'raw s_h',A['s_h']); report(c,'raw s_uv',A['s_uv'])
        else: print(c,'saved',len(G))
