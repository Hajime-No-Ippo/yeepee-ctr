"""LightGBM variants on the v12 feature matrix (dwbig-like). python xlgb.py CUT TAG [--seed 0] [--rounds 1000] [--save 1000] [--params json] [--group]
Same loader/folds/drop as v12 ens.py (cwd=/workspace/huawei/v12, cl from v12x). --group: ranking objectives, group=[10]*n_events (rows are event-major, verified).
Saves v12x/ens/{tag}_c{CUT}_s{seed}.npy (raw scores at --save rounds)."""
import sys, os, re, json, gc, time, argparse
sys.path.insert(0,'/workspace/huawei/v18classic')
import numpy as np, lightgbm as lgb
from cl import matrix
from sklearn.metrics import roc_auc_score
ap=argparse.ArgumentParser(); ap.add_argument('cut',type=int); ap.add_argument('tag'); ap.add_argument('--seed',type=int,default=0)
ap.add_argument('--rounds',type=int,default=1000); ap.add_argument('--save',default='1000'); ap.add_argument('--params',default='{}'); ap.add_argument('--group',action='store_true')
ap.add_argument('--threads',type=int,default=8); ap.add_argument('--lr',default='feat_lr2,feat_tt,feat_pp,feat_sqC,feat_dw')
a=ap.parse_args(); t0=time.time(); DROP='^sq_(pdot|pcos|mdot|mcos)$'; extra=tuple(x for x in a.lr.split(',') if x)
trn=[f'f{k}' for k in range(max(1,a.cut-24),a.cut-20)]
Xs=[];ys=[]
for n in trn:
    X,m,f=matrix(n,1.0,variant='notime',extra=extra); keep=[i for i,c in enumerate(f) if not re.search(DROP,c)]; fl=[f[i] for i in keep]
    Xs.append(np.ascontiguousarray(X[:,keep])); ys.append(m['clicked'].to_numpy()); del X,m; gc.collect()
X=np.concatenate(Xs); del Xs; y=np.concatenate(ys); gc.collect(); assert len(y)%10==0
P=dict(objective='binary',learning_rate=0.015,num_leaves=127,min_data_in_leaf=200,feature_fraction=0.5,bagging_fraction=0.8,bagging_freq=1,lambda_l2=30.0,
       max_bin=255,force_col_wise=True,verbose=-1,num_threads=a.threads,seed=42+a.seed); P.update(json.loads(a.params))
t_load=time.time()-t0; t1=time.time()
ds=lgb.Dataset(X,y,feature_name=fl,free_raw_data=True,group=[10]*(len(y)//10) if a.group else None,params={k:P[k] for k in ('max_bin','force_col_wise','max_bin_by_feature') if k in P}).construct(); t_ds=time.time()-t1; del X; gc.collect()
vn='eval' if a.cut==27 else f'val{a.cut}'
Xv,mv,fv=matrix(vn,1.0,variant='notime',extra=extra); Xv=Xv[:,[fv.index(c) for c in fl]]
RS=[int(r) for r in a.save.split(',')]; t1=time.time(); mdl=lgb.train(P,ds,max(RS)); t_fit=time.time()-t1
S=np.stack([mdl.predict(Xv,num_iteration=r,raw_score=True) for r in RS],1)
np.save(f'/workspace/huawei/v18classic/ens/{a.tag}_c{a.cut}_s{a.seed}.npy',S.astype(np.float32))
imp=dict(zip(fl,mdl.feature_importance('gain').tolist())); json.dump(imp,open(f'/workspace/huawei/v18classic/ens/{a.tag}_c{a.cut}_s{a.seed}_gain.json','w'))
msg=dict(cut=a.cut,tag=a.tag,seed=a.seed,n=len(y),nf=len(fl),load_s=round(t_load),ds_s=round(t_ds),fit_s=round(t_fit),min=round((time.time()-t0)/60,1),obj=P['objective'],max_bin=P['max_bin'])
if a.cut!=27: msg['auc']={r:round(roc_auc_score(mv['clicked'].to_numpy(),S[:,i]),5) for i,r in enumerate(RS)}
print(json.dumps(msg),flush=True); open('/workspace/huawei/v18classic/ens/xlog.jsonl','a').write(json.dumps(msg)+'\n')
