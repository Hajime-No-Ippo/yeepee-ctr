"""mkgf.py TABLE [TABLE..]: per-row graph features (7-seed avg) -> cache/gf_{table}.npy (float32, n_rows x 8) + cache/gf_names.json.
Table -> train-only graph cutoff k = fold cutoff (f{k}->k, val23->3, val25->5, val26->6, eval->7); embeddings trained on click graph ts<J20+k*DAY only. No labels/eval aggregates used.
Feature scales are stable across cutoffs (checked in drift.py) so raw values are used (no per-table normalisation)."""
import sys,os; sys.dont_write_bytecode=True; sys.path.insert(0,'/workspace/huawei/v21graph')
import numpy as np
from gfeat import feats,load_tab
OG='/workspace/huawei/v21graph/out/'; OL='/workspace/huawei/v21gfeat/out/'
GT=os.environ.get('GTAG','')
SEEDS=[int(x) for x in os.environ.get("GSEEDS","0,1,2,3,4,5,9").split(",")]
CFG={'f1':('u1b',1,800),'f2':('u1b',2,800),'f3':('u1b',3,800),'f4':('u1',4,350),'f5':('u1',5,350),'f6':('u1',6,350),
     'val23':('u1b',3,800),'val25':('u1',5,350),'val26':('u1',6,350),'eval':('u1',7,350)}
NAMES=['g_suv','g_sh','g_cosm','g_cosx','g_G','g_Gsd','g_udeg','g_ideg']
def embpath(tag,k,s):
    p=OG+f'emb_{tag}_k{k}_s{s}.npz'
    return p if os.path.exists(p) else OL+f'emb_{tag}_k{k}_s{s}.npz'
def build(nm):
    tag,k,st=CFG[nm]; acc={n:0. for n in ('s_uv','s_h','cos_mean','cos_max')}; Gs=[]
    for s in SEEDS:
        e=np.load(embpath(tag,k,s)); F=feats(nm,e[f'U{st}'],e[f'V{st}'])
        for n in acc: acc[n]=acc[n]+F[n].astype(np.float64)/len(SEEDS)
        Gs.append(F['s_uv']/1.47+F['s_h']/1.0+F['cos_mean']/0.11)
    G=np.mean(Gs,0); Gsd=np.std(Gs,0)
    g=np.load(OG+f'g_{k}.npz'); NU,NI=np.load(OG+'dims.npy'); key=np.unique(g['cu'].astype(np.int64)*NI+g['ci'].astype(np.int64)); ud=np.bincount(key//NI,minlength=NU+1); idg=np.bincount(key%NI,minlength=NI+1)
    T=load_tab(nm); u=np.minimum(T['user'],NU); ud_r=np.where(T['user']>=NU,0,ud[u]); id_r=idg[T['item']]
    X=np.stack([acc['s_uv'],acc['s_h'],acc['cos_mean'],acc['cos_max'],G,Gsd,np.log1p(ud_r),np.log1p(id_r)],1).astype(np.float32)
    np.save(f'/workspace/huawei/v21gfeat/cache/gf{GT}_{nm}.npy',X); np.save(f'/workspace/huawei/v21gfeat/cache/gf{GT}_{nm}_rowid.npy',T['rowid'])
    print(nm,X.shape,np.isfinite(X).all(),X.mean(0).round(3),flush=True)
if __name__=='__main__':
    import json; json.dump(NAMES,open('/workspace/huawei/v21gfeat/cache/gf_names.json','w'))
    for nm in sys.argv[1:]: build(nm)
