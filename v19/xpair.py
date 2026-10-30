"""Cross-slate pairwise LightGBM (RankNet-style) targeting GLOBAL ROC AUC. python xpair.py CUT TAG [--m 6] [--lr feat dirs] [--rounds 600] [--save 300,600] [--lrate 0.03]
Rows are event-major (10 per slate). Every boosting iteration the training events are randomly re-partitioned into groups of m slates (10*m rows);
loss = sum over (clicked p, non-clicked n) pairs inside a group of log(1+exp(-(s_p-s_n))) (all pairs weighted equally = AUC). m=1 -> pure within-slate; large m -> mostly cross-slate pairs
(share of within-slate pairs ~ 1/m). Same data/feature loader/folds/drop as xlgb17.py (dwbig recipe tree shape). Saves ens/{tag}_c{cut}_s{seed}.npy (raw scores at --save rounds)."""
import sys, os, re, json, gc, time, argparse
sys.path.insert(0,'/workspace/huawei/v19')
import numpy as np, lightgbm as lgb
from cl import matrix
from sklearn.metrics import roc_auc_score
ap=argparse.ArgumentParser(); ap.add_argument('cut',type=int); ap.add_argument('tag'); ap.add_argument('--seed',type=int,default=0)
ap.add_argument('--rounds',type=int,default=600); ap.add_argument('--save',default='300,600'); ap.add_argument('--params',default='{}'); ap.add_argument('--m',type=int,default=6)
ap.add_argument('--threads',type=int,default=8); ap.add_argument('--lr',default='feat_lr2,feat_tt,feat_pp,feat_sqC,feat_dw'); ap.add_argument('--lrate',type=float,default=0.03)
ap.add_argument('--negfrac',type=float,default=1.0)
a=ap.parse_args(); t0=time.time(); DROP='^sq_(pdot|pcos|mdot|mcos)$'; extra=tuple(x for x in a.lr.split(',') if x)
trn=[f'f{k}' for k in range(max(1,a.cut-24),a.cut-20)]
Xs=[];ys=[]
for n in trn:
    X,m,f=matrix(n,1.0,variant='notime',extra=extra); keep=[i for i,c in enumerate(f) if not re.search(DROP,c)]; fl=[f[i] for i in keep]
    Xs.append(np.ascontiguousarray(X[:,keep])); ys.append(m['clicked'].to_numpy()); del X,m; gc.collect()
X=np.concatenate(Xs); del Xs; y=np.concatenate(ys); gc.collect(); assert len(y)%10==0
NE=len(y)//10; rng=np.random.RandomState(1000+a.seed)
P=dict(objective='binary',learning_rate=a.lrate,num_leaves=127,min_data_in_leaf=200,feature_fraction=0.5,bagging_fraction=0.8,bagging_freq=1,lambda_l2=30.0,
       max_bin=255,force_col_wise=True,verbose=-1,num_threads=a.threads,seed=42+a.seed); P.update(json.loads(a.params))
G=10*a.m; ng=NE//a.m; used=ng*a.m
Y2=y.reshape(NE,10)
def pair_obj(preds,ds):
    perm=rng.permutation(NE)[:used]; ridx=(perm[:,None]*10+np.arange(10)[None,:]).reshape(ng,G)     # (ng,G) row indices
    S=preds[ridx]; Yg=Y2[perm].reshape(ng,G)
    gi,pi=np.nonzero(Yg==1)
    grad=np.zeros(len(y)); hess=np.full(len(y),1e-3)
    if len(gi)==0: return grad,hess
    d=S[gi,pi][:,None]-S[gi,:]                                  # (npos,G): s_p - s_j
    neg=(Yg[gi,:]==0)
    r=np.where(neg,1.0/(1.0+np.exp(np.clip(d,-30,30))),0.0)    # sigma(-(s_p-s_n)) on pos x neg pairs
    h=r*(1-r)
    npos=(Yg==1).sum(1); nneg=G-npos
    # per-row normalisation by the mean number of partners so that gradient scale ~ pointwise logloss
    z=1.0/max(1.0,G*float((y==1).mean())*(1-float((y==1).mean())))
    gp=-r.sum(1); hp=h.sum(1)
    flat_p=ridx[gi,pi]; grad[flat_p]+=gp; hess[flat_p]+=hp
    flat_n=(ridx[gi,:]).ravel(); nm=neg.ravel()
    N=len(y); wr=r.ravel()[nm]; wh=h.ravel()[nm]; fn=flat_n[nm]
    grad+=np.bincount(fn,weights=wr,minlength=N); hess+=np.bincount(fn,weights=wh,minlength=N)
    return grad*z,hess*z
P['objective']=pair_obj
t1=time.time()
ds=lgb.Dataset(X,y,feature_name=fl,free_raw_data=True,params={k:P[k] for k in ('max_bin','force_col_wise') if k in P}).construct(); t_ds=time.time()-t1; del X; gc.collect()
vn='eval' if a.cut==27 else f'val{a.cut}'
Xv,mv,fv=matrix(vn,1.0,variant='notime',extra=extra); Xv=Xv[:,[fv.index(c) for c in fl]]
RS=[int(r) for r in a.save.split(',')]; t1=time.time()
init=float(np.log(y.mean()/(1-y.mean())))
mdl=lgb.train(P,ds,max(RS)); t_fit=time.time()-t1
S=np.stack([mdl.predict(Xv,num_iteration=r,raw_score=True) for r in RS],1)
np.save(f'/workspace/huawei/v19/ens/{a.tag}_c{a.cut}_s{a.seed}.npy',S.astype(np.float32))
imp=dict(zip(fl,mdl.feature_importance('gain').tolist())); json.dump(imp,open(f'/workspace/huawei/v19/ens/{a.tag}_c{a.cut}_s{a.seed}_gain.json','w'))
msg=dict(cut=a.cut,tag=a.tag,seed=a.seed,m=a.m,n=len(y),nf=len(fl),ds_s=round(t_ds),fit_s=round(t_fit),min=round((time.time()-t0)/60,1),lrate=a.lrate)
if a.cut!=27: msg['auc']={r:round(roc_auc_score(mv['clicked'].to_numpy(),S[:,i]),5) for i,r in enumerate(RS)}
print(json.dumps(msg),flush=True); open('/workspace/huawei/v19/ens/xlog.jsonl','a').write(json.dumps(msg)+'\n')
