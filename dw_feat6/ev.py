import numpy as np,json,os
from sklearn.metrics import roc_auc_score as AUC
E='/workspace/huawei/v20/ens'; res={}
def z(a): a=a.astype(np.float64); return (a-a.mean())/a.std()
for c in (23,25):
    o=np.load(f'/workspace/huawei/cb_blend/oof_c{c}.npz'); y=o['y']; M=[z(o[k]) for k in('raw_z','raw_M','raw_big','raw_D')]
    refs={s:np.load(f'{E}/ref484_c{c}_s{s}.npy') for s in range(3)}; ar={s:AUC(y,r) for s,r in refs.items()}
    bl=lambda s: AUC(y,np.mean(M+[z(s)],0)); b4=AUC(y,np.mean(M,0)); br={s:bl(r) for s,r in refs.items()}
    d=dict(ref_single=ar,ref_mean=float(np.mean(list(ar.values()))),ref_sd=float(np.std(list(ar.values()),ddof=1)),blend4=b4,blend4_ref=br)
    d['ref_pair_diffs']=[ar[1]-ar[0],ar[2]-ar[0],ar[2]-ar[1]]; d['blendref_pair_diffs']=[br[1]-br[0],br[2]-br[0],br[2]-br[1]]
    for s in (0,1):
        f=f'out/f6_c{c}_s{s}.npy'
        if not os.path.exists(f): continue
        x=np.load(f); a=AUC(y,x); b=bl(x)
        d[f'f6_s{s}']=dict(single=a,d_single_same_seed=a-ar[s],d_single_vs_mean3=a-d['ref_mean'],blend=b,d_blend_same_seed=b-br[s],d_blend_vs_blend4=b-b4,corr_ref=float(np.corrcoef(x,refs[s])[0,1]))
    res[c]=d
json.dump(res,open('ev.json','w'),indent=1,default=float); print(json.dumps(res,indent=1,default=float))
