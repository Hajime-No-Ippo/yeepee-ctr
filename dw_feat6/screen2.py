"""per-column residual screen on pool (gf+prop), delta vs matched noise col (mean of 3 noise seeds); cut arg. 60 trees lr .05 on [col, col_dm] each."""
import sys,json,numpy as np,polars as pl,lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score as AUC
cut=int(sys.argv[1]); H='/workspace/huawei'
v=pl.read_parquet(f'{H}/v12/feat_v3/val{cut}.parquet',columns=['rowid','u','clicked']); y=v['clicked'].to_numpy(); u=v['u'].to_numpy()
base=np.load(f'{H}/v20/ens/ref484_c{cut}_s0.npy').astype(np.float64); a0=AUC(y,base)
folds=list(GroupKFold(3).split(base,y,u))
def dm(a):
    g=a.reshape(-1,10); 
    with np.errstate(all='ignore'): return (g-np.nanmean(g,1,keepdims=True)).ravel()
def run(X,n=60):
    p=np.zeros(len(y))
    for tr,te in folds:
        m=lgb.train(dict(objective='binary',learning_rate=.05,num_leaves=31,min_data_in_leaf=300,feature_fraction=1.0,bagging_fraction=.8,bagging_freq=1,lambda_l2=30,verbose=-1,num_threads=2,seed=1),lgb.Dataset(X[tr],y[tr],init_score=base[tr]),n)
        p[te]=base[te]+m.predict(X[te],raw_score=True)
    return AUC(y,p)-a0
noise=[run(np.random.RandomState(s).randn(len(y),2).astype(np.float32)) for s in range(3)]; print('noise',noise,flush=True)
res={'noise':noise}
for d in ('v12x/feat_s9_gf','v12x/feat_prop'):
    t=v.select('rowid').join(pl.read_parquet(f'{H}/{d}/val{cut}.parquet'),on='rowid',how='left',maintain_order='left')
    for c in t.columns[1:]:
        if c.endswith(('_dm','_z')): continue
        a=t[c].to_numpy().astype(np.float32); r=run(np.column_stack([a,dm(a)])); res[c]=r; print(c,round(r,5),flush=True)
        json.dump(res,open(f'screen2_c{cut}.json','w'),indent=1)
