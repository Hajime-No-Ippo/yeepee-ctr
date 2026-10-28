import sys, numpy as np, polars as pl, glob
sys.path.insert(0,'/workspace/huawei/dw_levers')
from common import z12, A, H
from scipy.stats import rankdata
for c in (25,23):
    y=pl.read_parquet(f'{H}/v10/feat_v3/val{c}.parquet',columns=['clicked']).to_series().to_numpy().astype(int)
    z=z12(c); print(f'== C{c} z12 {A(y,z):.5f}')
    M={}
    for f in sorted(glob.glob(f'{H}/v12/ens/*_c{c}_s*.npy')):
        a=np.load(f); nm=f.split('/')[-1][:-4]
        if a.ndim==1: a=a[:,None]
        for k in range(a.shape[1]): M[f'{nm}[{k}]']=a[:,k]
    for nm,a in M.items():
        if np.isnan(a).any() or len(a)!=len(y): continue
        r=A(y,a); 
        if r<0.7: continue
        # incremental: best w in z + w*zscore(a)
        zs=(a-a.mean())/a.std(); best=max((A(y,z+w*zs*z.std()),w) for w in (0,0.05,0.1,0.2,0.4))
        print(f'  {nm:18s} AUC {r:.5f} corr(z) {np.corrcoef(z,a)[0,1]:.3f} best blend {best[0]-A(y,z):+.5f} w={best[1]}')
