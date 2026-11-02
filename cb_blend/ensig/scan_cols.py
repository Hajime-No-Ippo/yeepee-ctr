import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ['k' if False else k]='4'
import numpy as np, polars as pl, glob
from sklearn.metrics import roc_auc_score
H='/workspace/huawei/'
yv={c:pl.read_parquet(f'{H}v12/feat_v3/val{c}.parquet',columns=['clicked'])['clicked'].to_numpy() for c in (23,25,26)}
pats={'z_b':'v19/ens/nmAs_c{c}_s0','z_s':'v19/ens/nmAs_s_c{c}_s0','z_mlp':'v19/ens/nmAsmlp_c{c}_s0','dwbig':'v12/ens/dwbig_c{c}_s0','dw_s':'v12/ens/dw_s_c{c}_s0','dwmlp':'v12/ens/dwmlp_c{c}_s0','mlp2':'v12/ens/mlp2_c{c}_s0',
'sC_s':'v12/ens/sC_s_c{c}_s0','sCbig':'v12/ens/sCbig_c{c}_s0','coAp':'v17cooc/ens/coAp_c{c}_s0','coBp':'v17cooc/ens/coBp_c{c}_s0','coA_s':'v17cooc/ens/coA_s_c{c}_s0','coAmlp':'v17cooc/ens/coAmlp_c{c}_s0',
'v2base':'v2work/ens/base_c{c}_s0','v2slow':'v2work/ens/slow_c{c}_s0','estrc':'v2work/ens/estrc_c{c}_s0','wide':'v2work/ens/wide_c{c}_s0','v6base':'v2work/ens/v6base_c{c}_s0','v7base':'v2work/ens/v7base_c{c}_s0',
'xgb':'v12x/ens/xgb_c{c}_s0','lr':'v9d/val/ens/lr_c{c}','ctl_b':'v9d/val/ens/ctl_b_c{c}_s0','ctl_s':'v9d/val/ens/ctl_s_c{c}_s0','listA':'v21diag/ens/v21listA_c{c}_s0','pslate':'v21diag/ens/pslate_c{c}_A','ctl6':'v21seq/ens/ctl6_A_c{c}_s0','big':'v21seq/ens/v21seq_big_A_c{c}_s0','lite':'lite/ens/base_c{c}_s0'}
for k,p in pats.items():
    for c in (23,25,26,27):
        f=H+p.format(c=c)+'.npy'
        if not os.path.exists(f): f=H+p.format(c=c)+'.npy'; 
        if not os.path.exists(f): print(k,c,'missing'); continue
        a=np.load(f,mmap_mode='r'); out=[]
        if c!=27 and a.shape[0]==len(yv[c]):
            cols=range(a.shape[1]) if a.ndim>1 else [None]
            for j in cols:
                v=np.asarray(a if j is None else a[:,j],dtype=np.float64); out.append(round(roc_auc_score(yv[c],v),4))
        print(k,c,a.shape,a.dtype,out,'' if (c==27 or a.shape[0]==len(yv[c])) else f'LEN MISMATCH vs {len(yv[c])}',flush=True)
