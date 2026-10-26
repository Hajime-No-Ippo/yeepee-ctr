"""Phase 1 ensemble components. python ens.py CUT TAG [--seeds 0,1] [--folds f|fe] [--stale] [--rc] [--drop re] [--params json] [--rounds 400]
CUT=25/26: train f1..f(CUT-21) (+e/d folds) full data, predict val{CUT}; CUT=27: train f1..f6 (+E1..E6), predict eval.
Saves ens/{tag}_c{CUT}_s{seed}.npy with raw scores at rounds 300,400(,500) (columns)."""
import sys, re, json, gc, time, argparse, numpy as np, lightgbm as lgb
from cl import matrix
from sklearn.metrics import roc_auc_score
ap=argparse.ArgumentParser(); ap.add_argument('cut',type=int); ap.add_argument('tag'); ap.add_argument('--seeds',default='0')
ap.add_argument('--folds',default='f'); ap.add_argument('--stale',action='store_true'); ap.add_argument('--rc',action='store_true')
ap.add_argument('--drop',default=None); ap.add_argument('--params',default='{}'); ap.add_argument('--rounds',type=int,default=400); ap.add_argument('--threads',type=int,default=5); ap.add_argument('--lr',default=''); ap.add_argument('--nfolds',type=int,default=99); ap.add_argument('--negfrac',type=float,default=1.0); ap.add_argument('--keep',default='')
a=ap.parse_args(); t0=time.time()
extra=(('feat_rc',) if a.rc else ())+tuple(x for x in a.lr.split(',') if x); var='stale' if a.stale else 'notime'
efold={23:[],25:[f'e{k}' for k in range(1,5)],26:[f'd{k}' for k in range(1,6)],27:[f'E{k}' for k in range(1,7)]}[a.cut]
trn=([f'f{k}' for k in range(max(1,a.cut-20-a.nfolds),a.cut-20)] if 'f' in a.folds else [])+(efold if 'e' in a.folds else [])
Xs=[];ys=[];feats=None
for n in trn:
    X,m,f=matrix(n,1.0,variant=var,extra=extra); feats=f; Xs.append(X); ys.append(m['clicked'].to_numpy()); del X,m; gc.collect()
X=np.concatenate(Xs); del Xs; gc.collect(); y=np.concatenate(ys)
keep=[i for i,c in enumerate(feats) if not (a.drop and re.search(a.drop,c))]; fl=[feats[i] for i in keep]
if a.keep: KS=set(open(a.keep).read().split()); keep=[i for i in keep if feats[i] in KS]; fl=[feats[i] for i in keep]
if a.negfrac<1:
    rs=np.random.RandomState(123); sel=(y==1)|(rs.rand(len(y))<a.negfrac); X=X[sel]; y=y[sel]
if len(keep)<len(feats): X=np.ascontiguousarray(X[:,keep])
P=dict(objective='binary',learning_rate=0.05,num_leaves=63,min_data_in_leaf=200,feature_fraction=0.7,bagging_fraction=0.8,
       bagging_freq=1,lambda_l2=10.0,verbose=-1,num_threads=a.threads); P.update(json.loads(a.params))
t_load=time.time()-t0; t_ds=time.time(); ds=lgb.Dataset(X,y,feature_name=fl,free_raw_data=True,params={k:P[k] for k in ('max_bin','force_col_wise') if k in P}).construct(); t_ds=time.time()-t_ds; del X; gc.collect()
vn='eval' if a.cut==27 else f'val{a.cut}'
Xv,mv,fv=matrix(vn,1.0,variant=var,extra=extra); Xv=Xv[:,[fv.index(c) for c in fl]]
RS=[r for r in (300,400,500) if r<=a.rounds]
for s in map(int,a.seeds.split(',')):
    t1=time.time(); mdl=lgb.train(dict(P,seed=42+s),ds,a.rounds); t_fit=time.time()-t1; t1=time.time()
    S=np.stack([mdl.predict(Xv,num_iteration=r,raw_score=True) for r in RS],1)
    t_pred=time.time()-t1; np.save(f'ens/{a.tag}_c{a.cut}_s{s}.npy',S.astype(np.float32))
    mdl.save_model(f'ens/{a.tag}_c{a.cut}_s{s}.txt')
    msg=dict(cut=a.cut,tag=a.tag,seed=s,n=len(y),nf=len(fl),min=round((time.time()-t0)/60,1),load_s=round(t_load),ds_s=round(t_ds),fit_s=round(t_fit),pred_s=round(t_pred))
    if a.cut!=27: msg['auc']={r:round(roc_auc_score(mv['clicked'].to_numpy(),S[:,i]),5) for i,r in enumerate(RS)}
    print(json.dumps(msg),flush=True); open('ens/ens_log.jsonl','a').write(json.dumps(msg)+'\n')
