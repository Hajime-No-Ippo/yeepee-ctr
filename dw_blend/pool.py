import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import numpy as np, polars as pl, glob, ast, re, json
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score as AUC
H='/workspace/huawei/'
Y={c:pl.read_parquet(f'{H}v12/feat_v3/val{c}.parquet',columns=['clicked'])['clicked'].to_numpy() for c in (23,25)}
rk=lambda x:((rankdata(x)-1)/(len(x)-1)).astype(np.float32)
# name -> (glob template with {c}, has_c27)
POOL={
 'dwbig':'v12/ens/dwbig_c{c}_s*.npy','dwmlp':'v12/ens/dwmlp_c{c}_s*.npy','mlp2':'v12/ens/mlp2_c{c}_s*.npy','sCbig':'v12/ens/sCbig_c{c}_s*.npy','dw_s':'v12/ens/dw_s_c{c}_s*.npy',
 'xgb':'v12x/ens/xgb_c{c}_s*.npy','f_all':'v12x/ens/f_all_c{c}_s*.npy','lrank':'v12x/ens/lrank_c{c}_s*.npy','xendcg':'v12x/ens/xendcg_c{c}_s*.npy',
 'v10A':'v10/ens/v10A_blend_c{c}.npy','sqs':'v8/ens/sqs_blend_c{c}.npy','tA':'v9/ens/tA_blend_c{c}.npy','lr9d':'v9d/val/ens/lr_c{c}.npy',
 'coAp':'v17cooc/ens/coAp_c{c}_s*.npy','coBp':'v17cooc/ens/coBp_c{c}_s*.npy','v18fr':'v18classic/ens/v18fr_c{c}_s*.npy','nmAp':'v19/ens/nmAp_c{c}_s*.npy','nmBs':'v19/ens/nmBs_c{c}_s*.npy',
 'dcn_cr64':'v21dcn/ens/r1_cr64_A_c{c}_s*.npy','dcn_a1':'v21dcn/ens/r1_a1_A_c{c}_s*.npy','listB':'v21diag/ens/v21listB_c{c}_s*.npy','ctl6B':'v21seq/ens/ctl6_B_c{c}_s*.npy',
}
def load(pat,c):
    fs=sorted(glob.glob(H+pat.format(c=c))); assert fs,(pat,c)
    out=[]
    for f in fs:
        a=np.asarray(np.load(f,mmap_mode='r'),dtype=np.float64)
        if a.ndim==1: a=a[:,None]
        if a.shape[0]!=len(Y[c]): continue
        # pick the column with best AUC (output head); same column index reused across seeds
        out.append(a)
    cols=out[0].shape[1]; best=max(range(cols),key=lambda j:AUC(Y[c],out[0][:,j]))
    r=np.mean([rk(o[:,best]) for o in out],0)   # seed-average in rank space
    return r,len(out),best
R={c:{} for c in (23,25)}; info=[]
for n,pat in POOL.items():
    for c in (23,25):
        try:
            r,k,b=load(pat,c); R[c][n]=r; info.append((n,c,k,b,round(AUC(Y[c],r),5)))
        except Exception as e: info.append((n,c,0,-1,str(e)[:40]))
    print(n,[i for i in info if i[0]==n],flush=True)
# add cb_blend members
for c in (23,25):
    z=np.load(f'{H}cb_blend/oof_c{c}.npz',allow_pickle=True)
    for m in ['z','M','big','D','ref','A','list1']:
        if f'raw_{m}' in z.files: R[c][m]=rk(z[f'raw_{m}'].astype(np.float64))
common=[m for m in R[23] if m in R[25]]
np.savez('pool_c23.npz',**{m:R[23][m] for m in common}); np.savez('pool_c25.npz',**{m:R[25][m] for m in common})
json.dump(info,open('pool_info.json','w'))
print(common)
