"""XGBoost (hist, CPU) on the exact v12 feature matrix/folds used by dwbig. Run with cwd=/workspace/huawei/v12 (uses its cl.matrix + feat dirs).
python xens.py CUT TAG [--pick] [--rounds N] [--save r1,r2] [--seed 0] [--params json]
--pick : train on all training folds except the LAST one, early-stop on the last training fold (never the val cut); prints best iteration, no val predictions.
else   : train on all training folds (same as dwbig: f(max(1,cut-24))..f(cut-21), all negatives, drop sq_(pdot|pcos|mdot|mcos)), save raw margins at --save rounds to v12x/ens/{tag}_c{CUT}_s{seed}.npy"""
import sys, os, re, json, gc, time, argparse
sys.path.insert(0,'/workspace/huawei/v12x/pylib'); sys.path.insert(0,'/workspace/huawei/v12x')
import numpy as np, polars as pl, xgboost as xgb
from cl import matrix
from sklearn.metrics import roc_auc_score
ap=argparse.ArgumentParser(); ap.add_argument('cut',type=int); ap.add_argument('tag'); ap.add_argument('--pick',action='store_true')
ap.add_argument('--rounds',type=int,default=1500); ap.add_argument('--save',default='500'); ap.add_argument('--seed',type=int,default=0)
ap.add_argument('--params',default='{}'); ap.add_argument('--threads',type=int,default=8); ap.add_argument('--lr',default='feat_lr2,feat_tt,feat_pp,feat_sqC,feat_dw')
a=ap.parse_args(); t0=time.time(); DROP='^sq_(pdot|pcos|mdot|mcos)$'
extra=tuple(x for x in a.lr.split(',') if x)
trn=[f'f{k}' for k in range(max(1,a.cut-24),a.cut-20)]
def load(names):  # +-inf -> +-1e15 (xgboost rejects inf); NaN kept as missing
    Xs=[];ys=[];fl=None
    for n in names:
        X,m,f=matrix(n,1.0,variant='notime',extra=extra); keep=[i for i,c in enumerate(f) if not re.search(DROP,c)]; fl=[f[i] for i in keep]
        Xs.append(np.clip(np.ascontiguousarray(X[:,keep]),-1e15,1e15)); ys.append(m['clicked'].to_numpy()); del X,m; gc.collect()
    return np.concatenate(Xs),np.concatenate(ys),fl
P=dict(objective='binary:logistic',tree_method='hist',grow_policy='lossguide',max_leaves=127,max_depth=0,eta=0.03,subsample=0.8,colsample_bytree=0.5,
       min_child_weight=30,reg_lambda=30,max_bin=255,nthread=a.threads,seed=42+a.seed,eval_metric='auc'); P.update(json.loads(a.params))
if a.pick:
    X,y,fl=load(trn[:-1]); Xh,yh,_=load(trn[-1:])
    dtr=xgb.QuantileDMatrix(X,y,feature_names=fl,max_bin=P['max_bin']); del X; dh=xgb.DMatrix(Xh,yh,feature_names=fl); del Xh; gc.collect()
    ev={}; b=xgb.train(P,dtr,a.rounds,evals=[(dh,'hold')],early_stopping_rounds=100,evals_result=ev,verbose_eval=50)
    r=dict(pick=True,cut=a.cut,train=trn[:-1],hold=trn[-1],best_iter=b.best_iteration,best_auc=b.best_score,fit_s=round(time.time()-t0)); print(json.dumps(r)); open('/workspace/huawei/v12x/ens/xlog.jsonl','a').write(json.dumps(r)+'\n'); sys.exit()
X,y,fl=load(trn); t_load=time.time()-t0
dtr=xgb.QuantileDMatrix(X,y,feature_names=fl,max_bin=P['max_bin']); del X; gc.collect()
vn='eval' if a.cut==27 else f'val{a.cut}'
Xv,mv,fv=matrix(vn,1.0,variant='notime',extra=extra); Xv=np.clip(Xv[:,[fv.index(c) for c in fl]],-1e15,1e15); dv=xgb.DMatrix(Xv,feature_names=fl)
RS=[int(r) for r in a.save.split(',')]; assert max(RS)<=a.rounds
t1=time.time(); b=xgb.train({k:v for k,v in P.items() if k!='eval_metric'},dtr,max(RS)); t_fit=time.time()-t1
S=np.stack([b.predict(dv,iteration_range=(0,r),output_margin=True) for r in RS],1)
np.save(f'/workspace/huawei/v12x/ens/{a.tag}_c{a.cut}_s{a.seed}.npy',S.astype(np.float32))
msg=dict(cut=a.cut,tag=a.tag,seed=a.seed,n=len(y),nf=len(fl),load_s=round(t_load),fit_s=round(t_fit),min=round((time.time()-t0)/60,1),params={k:P[k] for k in P if k not in('nthread',)})
if a.cut!=27: msg['auc']={r:round(roc_auc_score(mv['clicked'].to_numpy(),S[:,i]),5) for i,r in enumerate(RS)}
print(json.dumps(msg),flush=True); open('/workspace/huawei/v12x/ens/xlog.jsonl','a').write(json.dumps(msg)+'\n')
