"""ev3.py tag cut(23|25|26) seeds(comma) [step]: score variants of gcn3 embeddings; gain over base+G8 (tuned w) and over base."""
import sys,os,numpy as np; sys.dont_write_bytecode=True
sys.path.insert(0,'/workspace/huawei/v21graph2')
from ana import *   # evalg, gfeat, zs, feats
O3='/workspace/huawei/v21graph3/out/'
KO={23:3,25:5,26:6}; TB={23:'val23',25:'val25',26:'val26'}
def scores(tag,c,seed,step=650):
    e=np.load(O3+f'emb_{tag}_k{KO[c]}_s{seed}.npz'); U,V=e[f'U{step}'],e[f'V{step}']
    F=feats(TB[c],U,V); T=load_tab(TB[c]); NU=len(U)
    u=np.minimum(T['user'],NU-1); it=T['item']
    out={k:F[k] for k in ('s_uv','s_h','cos_mean')}
    out['bi']=e[f'bi{step}'][it]; out['bu']=np.where(T['user']<NU,e[f'bu{step}'][u],0)
    return out,T['y']
def comb(S,w): return sum(wi*zs(S[k]) for k,wi in w.items())
def gain(c,G,w8=None):
    y,B=bases(c); G=nr(G); b=B['base']; G8=nr(np.load(f'/workspace/huawei/v21graph2/out/graph2s3b_f6_c{c}.npy'))
    a0=AUC(y,b); ws=[.1,.15,.2,.25,.3]; g8=[AUC(y,(1-w)*b+w*G8) for w in ws]; i=int(np.argmax(g8)); w8=ws[i]; B2=(1-w8)*b+w8*G8; a2=AUC(y,B2)
    r=[(v,(AUC(y,(1-v)*B2+v*G)-a2)*1e5) for v in (.02,.05,.1,.15,.2,.3)]
    r0=[(v,(AUC(y,(1-v)*b+v*G)-a0)*1e5) for v in (.1,.2,.3)]
    return dict(auc=AUC(y,G),a_base=a0,a_G8=a2,w8=w8,g8=(a2-a0)*1e5,on_G8=r,on_base=r0)
if __name__=='__main__':
    tag=sys.argv[1]; c=int(sys.argv[2]); seeds=[int(s) for s in sys.argv[3].split(',')]
    acc=None
    for s in seeds:
        S,y=scores(tag,c,s); acc=S if acc is None else {k:acc[k]+S[k] for k in S}
    acc={k:v/len(seeds) for k,v in acc.items()}
    V={'s3b':{'s_uv':2,'cos_mean':1},'s3b+bi':{'s_uv':2,'cos_mean':1,'bi':1},'s3b+bi+bu':{'s_uv':2,'cos_mean':1,'bi':1,'bu':1},'uv':{'s_uv':1},'uv+bi+bu':{'s_uv':1,'bi':1,'bu':1},'bi':{'bi':1},'bu':{'bu':1},'bi+bu':{'bi':1,'bu':1},'s3b+.5(bi+bu)':{'s_uv':2,'cos_mean':1,'bi':.5,'bu':.5}}
    for n,w in V.items():
        G=comb(acc,w); r=gain(c,G); print(f'C{c} {tag} {n:16s} auc {r["auc"]:.5f} base {r["a_base"]:.5f} +G8@{r["w8"]} {r["g8"]:+.1f}e-5 | on_G8 '+' '.join(f'v{v}:{g:+.1f}' for v,g in r['on_G8'])+' | on_base '+' '.join(f'v{v}:{g:+.1f}' for v,g in r['on_base']),flush=True)
