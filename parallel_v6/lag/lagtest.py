"""Lag-1d test for v6. Train v6base config (ens.py 17:58 params, seed 42, 400r, 3 threads) on f1..f4 from v2work cached matrices (read-only);
predict val25 (cached, as v6 run), val25repro (rebuilt here from copied code, C=Jan25) and val25lag (same targets, all stats frozen at Jan24)."""
import os, sys, json, hashlib, gc, numpy as np, polars as pl, lightgbm as lgb
sys.path.insert(0,'/workspace/huawei/parallel/src_snapshot/v2work'); sys.path.insert(1,'/workspace/huawei/v3')
from feats import DROP
from sklearn.metrics import roc_auc_score
sys.path.insert(0,'/workspace/huawei/v4'); from common import report
D='/workspace/huawei/v2work'; L1='/workspace/huawei/parallel/lag'; L='/workspace/huawei/parallel_v6/lag'
def cpath(n):
    key=hashlib.md5(json.dumps([n,1.0,'notime',['feat_lr2'],os.path.getmtime(f'{D}/feat_v3/{n}.parquet')]+['feat_xf']).encode()).hexdigest()[:12]
    return f'{D}/cache_mx/{n}_{key}.parquet'
fl=json.load(open(cpath('val25')+'.json'))
Xs=[];ys=[]
for k in range(1,5):
    fp=cpath(f'f{k}'); assert json.load(open(fp+'.json'))==fl
    t=pl.read_parquet(fp); Xs.append(t.select(fl).to_numpy().astype(np.float32)); ys.append(t['clicked'].to_numpy()); del t; gc.collect()
X=np.concatenate(Xs); y=np.concatenate(ys); del Xs; gc.collect(); print('train',X.shape,flush=True)
P=dict(objective='binary',learning_rate=0.05,num_leaves=63,min_data_in_leaf=200,feature_fraction=0.7,bagging_fraction=0.8,
       bagging_freq=1,lambda_l2=10.0,verbose=-1,num_threads=3,seed=42)
ds=lgb.Dataset(X,y,feature_name=fl,free_raw_data=True).construct(); del X; gc.collect()
m=lgb.train(P,ds,400); m.save_model(f'{L}/model_v6base_f1to4_s0.txt'); del ds; gc.collect()
def assemble(v3,v2,xf,x2,lr):
    a=pl.read_parquet(v3); b=pl.read_parquet(v2); f2=[c for c in b.columns if c not in DROP]
    a=a.join(b.select(['rowid']+f2).rename({c:'v2_'+c for c in f2}),on='rowid',how='left')
    for p in (xf,x2,lr): a=a.join(pl.read_parquet(p),on='rowid',how='left')
    miss=[c for c in fl if c not in a.columns]; assert not miss, miss
    return a
sets={'val25 (v2work cache, as v6 run)':pl.read_parquet(cpath('val25')),
      'val25repro (xf+lr2 rebuilt here, C=Jan25)':assemble(f'{D}/feat_v3/val25.parquet',f'{D}/feat/val25.parquet',f'{L}/feat_xf/val25.parquet',f'{D}/feat_x2/val25.parquet',f'{L}/feat_lr2/val25.parquet'),
      'val25lag (all stats frozen at Jan24)':assemble(f'{L1}/feat_v3/val25lag.parquet',f'{L1}/feat/val25lag.parquet',f'{L}/feat_xf/val25lag.parquet',f'{L1}/feat_x2/val25lag.parquet',f'{L}/feat_lr2/val25lag.parquet')}
res={'n_feats':len(fl),'train_rows':int(len(y))}
ref=None
for nm,t in sets.items():
    t=t.sort('rowid') if ref is None else t.join(ref.select('rowid'),on='rowid',how='right').select(t.columns)
    if ref is None: ref=t.select('rowid','eid','clicked','tm_stale')
    assert (t['rowid']==ref['rowid']).all() and (t['clicked']==ref['clicked']).all()
    Xv=t.select(fl).to_numpy().astype(np.float32); p=m.predict(Xv,num_iteration=400)
    r=report(ref,p); r['auc300']=round(roc_auc_score(ref['clicked'].to_numpy(),m.predict(Xv,num_iteration=300)),5); res[nm]=r
    pl.DataFrame({'rowid':ref['rowid'],'p':p}).write_parquet(f'{L}/pred_{nm.split()[0]}.parquet'); print(nm,json.dumps(r),flush=True)
    # lr feature diffs vs cache
    if 'repro' in nm:
        c=sets['val25 (v2work cache, as v6 run)'].sort('rowid'); dd={}
        for col in [x for x in fl if x.startswith(('lr','x_'))]:
            a_=c[col].cast(pl.Float64).to_numpy(); b_=t[col].cast(pl.Float32).cast(pl.Float64).to_numpy(); ok=~(np.isnan(a_)|np.isnan(b_))
            dd[col]=float(np.abs(a_[ok]-b_[ok]).max()) if ok.any() else None
        res['repro_maxabsdiff_xf_lr2']={'n_cols':len(dd),'n_nonzero(>1e-5)':sum(1 for v in dd.values() if v and v>1e-5),'max':max(v for v in dd.values() if v is not None),
                                      'worst':sorted(dd.items(),key=lambda kv:-(kv[1] or 0))[:5]}
        print(res['repro_maxabsdiff_xf_lr2'],flush=True)
json.dump(res,open(f'{L}/lagtest.json','w'),indent=1)
