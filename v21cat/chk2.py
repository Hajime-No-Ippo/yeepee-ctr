import numpy as np,json,sys
sys.path.insert(0,'/workspace/huawei/v21seq'); from evalseq import yv, AUC
nm=json.load(open('cache/names.json')); X=np.load('cache/val25.npy'); y=yv(25)
r=[]
for i,n in enumerate(nm):
    x=X[:,i].reshape(-1,10); x=np.where(np.isnan(x),np.nanmean(x) if not np.isnan(x).all() else 0,x); d=(x-x.mean(1,keepdims=True)).ravel()
    r.append((AUC(y,d),n))
for a,n in sorted(r,reverse=True)[:25]: print(f'{n:14s} {a:.4f}')
