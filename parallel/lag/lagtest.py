"""Train v2work's combo config (notime, XF=1, XF2=1, f1..f4, PARAMS from snapshot combo.py, 400 rounds, 3 threads) on v2work's
on-disk tables (read-only). Then predict (a) val25 as built and (b) val25lag (the same target rows, all stats frozen at Jan24).
Preds are saved in parallel/lag/."""
import sys, os, json, gc, numpy as np, polars as pl, lightgbm as lgb
sys.path.insert(0,'/workspace/huawei/parallel/src_snapshot/v2work'); sys.path.insert(1,'/workspace/huawei/v3')
from feats import DROP
from features import feature_cols, GROUPS
from sklearn.metrics import roc_auc_score
V='/workspace/huawei/v2work'; L='/workspace/huawei/parallel/lag'
V2_OF={'f1':'h1','f2':'h2','f3':'h3','f4':'h4','val25':'val25','val25lag':'val25lag'}
TIME=['tm_stale','tm_hour','tm_tb','tm_dt','v2_tb','v2_dt','v2_stale','v2_hour']
def load(n, base):
    a=pl.read_parquet(f'{base}/feat_v3/{n}.parquet'); b=pl.read_parquet(f'{base}/feat/{V2_OF[n]}.parquet')
    f2=[c for c in b.columns if c not in DROP]; b=b.select(['rowid']+f2).rename({c:'v2_'+c for c in f2})
    a=a.join(b,on='rowid',how='left')
    a=a.join(pl.read_parquet(f'{base}/feat_x/{n}.parquet'),on='rowid',how='left')
    return a.join(pl.read_parquet(f'{base}/feat_x2/{n}.parquet'),on='rowid',how='left')
PARAMS=dict(objective='binary',learning_rate=0.05,num_leaves=63,min_data_in_leaf=200,feature_fraction=0.7,bagging_fraction=0.8,
            bagging_freq=1,lambda_l2=10.0,verbose=-1,seed=42,num_threads=3,deterministic=True)
feats=None; Xs=[]; ys=[]
for k in range(1,5):
    t=load(f'f{k}',V)
    if feats is None:
        feats=[c for c in feature_cols(t,GROUPS)+[c for c in t.columns if c.startswith(('v2_','x_','r_','s_'))] if c not in TIME]
    Xs.append(t.select(feats).to_numpy().astype(np.float32)); ys.append(t['clicked'].to_numpy()); del t; gc.collect()
X=np.concatenate(Xs); del Xs; gc.collect()
print('train',X.shape,flush=True)
ds=lgb.Dataset(X,np.concatenate(ys),feature_name=feats,free_raw_data=True).construct(); del X; gc.collect()
m=lgb.train(PARAMS,ds,num_boost_round=400); del ds; gc.collect(); m.save_model(f'{L}/model_f1to4.txt')
res={'n_feats':len(feats)}
for n,base in (('val25',V),('val25lag',L)):
    va=load(n,base); miss=[c for c in feats if c not in va.columns]; assert not miss, miss
    Xv=va.select(feats).to_numpy().astype(np.float32); y=va['clicked'].to_numpy()
    res[n]={k:round(roc_auc_score(y,m.predict(Xv,num_iteration=k)),5) for k in (200,300,400)}
    pl.DataFrame({'rowid':va['rowid'],'p':m.predict(Xv,num_iteration=400)}).write_parquet(f'{L}/pred_{n}.parquet')
    print(n,res[n],flush=True)
json.dump(res,open(f'{L}/lagtest.json','w'),indent=1)
