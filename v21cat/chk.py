import numpy as np,json,sys
sys.path.insert(0,'/workspace/huawei/v21seq'); from evalseq import yv, AUC
nm=json.load(open('cache/names.json')); X=np.load('cache/val25.npy'); y=yv(25); print(X.shape,len(y))
for i,n in enumerate(nm):
    x=np.nan_to_num(X[:,i],nan=np.nanmedian(X[:,i])); a=AUC(y,x); print(f'{n:14s} {a:.4f} nan={np.isnan(X[:,i]).mean():.2f}')
