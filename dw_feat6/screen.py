"""Residual screen of cached, not-in-484 feature groups. base = ref484 s0 val logits (v20/ens), LightGBM on init_score, GroupKFold(user, 3), 2 threads.
usage: screen.py CUT  -> screen_c{CUT}.json  (delta AUC vs control(no extra cols=base) per group; also per-column gain)"""
import sys,os,json,time,numpy as np,polars as pl,lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score as AUC
cut=int(sys.argv[1]); H='/workspace/huawei'
v=pl.read_parquet(f'{H}/v12/feat_v3/val{cut}.parquet',columns=['rowid','u','clicked']); y=v['clicked'].to_numpy(); u=v['u'].to_numpy()
base=np.load(f'{H}/v20/ens/ref484_c{cut}_s0.npy').astype(np.float64); assert len(base)==len(y); print('base AUC',AUC(y,base),flush=True)
ref=set(json.load(open(f'{H}/v20/feats_ref484.json')))
G={'A_x':'v17cooc/feat_A','NA_x':'v19/feat_NA','prop':'v12x/feat_prop','tm':'v12x/feat_tm','B':'v17cooc/feat_B','NB':'v19/feat_NB','gf':'v12x/feat_s9_gf','tmB2':'v12/feat_tmB2'}
def dm(a): g=a.reshape(-1,10); return (g-np.nanmean(g,1,keepdims=True)).ravel()
def load(d):
    t=v.select('rowid').join(pl.read_parquet(f'{H}/{d}/val{cut}.parquet'),on='rowid',how='left',maintain_order='left'); cols=[c for c in t.columns[1:] if c not in ref]
    return {c:t[c].to_numpy().astype(np.float32) for c in cols}
folds=list(GroupKFold(3).split(base,y,u)); 
def run(X,n=150):
    p=np.zeros(len(y))
    for tr,te in folds:
        m=lgb.train(dict(objective='binary',learning_rate=.05,num_leaves=31,min_data_in_leaf=300,feature_fraction=.7,bagging_fraction=.8,bagging_freq=1,lambda_l2=30,verbose=-1,num_threads=2,seed=1),lgb.Dataset(X[tr],y[tr],init_score=base[tr]),n)
        p[te]=base[te]+m.predict(X[te],raw_score=True)
    return AUC(y,p),m
res={}
a0=AUC(y,base)
# control: noise column
rng=np.random.RandomState(0); a,_=run(rng.randn(len(y),1).astype(np.float32)); res['noise']=a-a0; print('noise',a-a0,flush=True)
for k,d in G.items():
    t=time.time(); F=load(d); names=list(F); X=np.column_stack([F[c] for c in names]+[dm(F[c]) for c in names])
    a,m=run(X); gain=dict(zip([c for c in names]+[c+'_dm' for c in names],m.feature_importance('gain')))
    res[k]=dict(delta=a-a0,n=len(names),top=sorted(gain,key=gain.get,reverse=True)[:8]); print(k,res[k],round(time.time()-t),flush=True)
    json.dump(res,open(f'screen_c{cut}.json','w'),indent=1,default=float)
