"""LightGBM component of the v12 recipe. python ens.py CUT TAG [--seeds 0] [--drop re] [--params json] [--rounds N] [--save r1,r2] [--lr dirs] [--nfolds 4] [--negfrac f]
CUT=27: train f(CUT-24)..f(CUT-21) (f3..f6), predict eval; CUT=25: train f1..f4, predict val25 (AUC printed).
Negatives of the concatenated training folds are subsampled with one RandomState(123).rand(total) draw. Saves ens/{tag}_c{CUT}_s{seed}.npy (raw scores at the --save rounds)."""
import sys, os, re, json, gc, time, argparse, numpy as np, lightgbm as lgb, polars as pl
from cl import matrix
from sklearn.metrics import roc_auc_score
ap=argparse.ArgumentParser(); ap.add_argument('cut',type=int); ap.add_argument('tag'); ap.add_argument('--seeds',default='0')
ap.add_argument('--drop',default=None); ap.add_argument('--params',default='{}'); ap.add_argument('--rounds',type=int,default=400); ap.add_argument('--threads',type=int,default=8)
ap.add_argument('--lr',default=''); ap.add_argument('--nfolds',type=int,default=4); ap.add_argument('--negfrac',type=float,default=1.0); ap.add_argument('--save',default='400')
a=ap.parse_args(); t0=time.time(); os.makedirs('ens',exist_ok=True)
extra=tuple(x for x in a.lr.split(',') if x)
trn=[f'f{k}' for k in range(max(1,a.cut-20-a.nfolds),a.cut-20)]
lens=[len(pl.read_parquet(f'feat_v3/{n}.parquet',columns=['rowid'])) for n in trn]
R=np.random.RandomState(123).rand(sum(lens)) if a.negfrac<1 else None
Xs=[];ys=[];feats=None;off=0
for n,L in zip(trn,lens):
    X,m,f=matrix(n,1.0,extra=extra); feats=f; yy=m['clicked'].to_numpy(); assert len(yy)==L
    sel=(yy==1)|(R[off:off+L]<a.negfrac) if R is not None else np.ones(L,bool); off+=L
    Xs.append(X[sel]); ys.append(yy[sel]); del X,m; gc.collect()
X=np.concatenate(Xs); del Xs; gc.collect(); y=np.concatenate(ys)
keep=[i for i,c in enumerate(feats) if not (a.drop and re.search(a.drop,c))]; fl=[feats[i] for i in keep]
if len(keep)<len(feats): X=np.ascontiguousarray(X[:,keep])
P=dict(objective='binary',learning_rate=0.05,num_leaves=63,min_data_in_leaf=200,feature_fraction=0.7,bagging_fraction=0.8,
       bagging_freq=1,lambda_l2=10.0,verbose=-1,num_threads=a.threads); P.update(json.loads(a.params))
t_load=time.time()-t0; t_ds=time.time(); ds=lgb.Dataset(X,y,feature_name=fl,free_raw_data=True,params={k:P[k] for k in ('max_bin','force_col_wise') if k in P}).construct(); t_ds=time.time()-t_ds; del X; gc.collect()
vn='eval' if a.cut==27 else f'val{a.cut}'
Xv,mv,fv=matrix(vn,1.0,extra=extra); Xv=Xv[:,[fv.index(c) for c in fl]]
RS=[int(r) for r in a.save.split(',') if int(r)<=a.rounds]
for s in map(int,a.seeds.split(',')):
    t1=time.time(); mdl=lgb.train(dict(P,seed=42+s),ds,a.rounds); t_fit=time.time()-t1; t1=time.time()
    S=np.stack([mdl.predict(Xv,num_iteration=r,raw_score=True) for r in RS],1)
    t_pred=time.time()-t1; np.save(f'ens/{a.tag}_c{a.cut}_s{s}.npy',S.astype(np.float32))
    msg=dict(cut=a.cut,tag=a.tag,seed=s,n=len(y),nf=len(fl),min=round((time.time()-t0)/60,1),load_s=round(t_load),ds_s=round(t_ds),fit_s=round(t_fit),pred_s=round(t_pred))
    if a.cut!=27: msg['auc']={r:round(roc_auc_score(mv['clicked'].to_numpy(),S[:,i]),5) for i,r in enumerate(RS)}
    print(json.dumps(msg),flush=True); open('ens/ens_log.jsonl','a').write(json.dumps(msg)+'\n')
