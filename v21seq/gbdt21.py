"""gbdt21.py CUT TAG --var A|B [--obj lambdarank|rank_xendcg|binary] [--rel K] [--rounds N --save r1,r2] [--params json] [--threads 8] [--seed 0]
Listwise LightGBM, slate=group (10 rows/event, event-major), on the v21list matrix (feat_lr2,tt,pp,sqC,dw,{A|B}p,N{A|B}s) + slate-relative features for the top-K gain features
(x - slate nanmean, x - slate max, within-slate rank pct). Variant B = transductive eval-history graph features (flag!). Saves ens/{TAG}_{var}_c{cut}_s{seed}.npy raw scores."""
import os,sys,re,json,gc,time,argparse
sys.path.insert(0,'/workspace/huawei/v19'); os.chdir('/workspace/huawei/v21seq'); sys.dont_write_bytecode=True
import numpy as np, lightgbm as lgb
from cl import matrix
from sklearn.metrics import roc_auc_score
ap=argparse.ArgumentParser(); ap.add_argument('cut',type=int); ap.add_argument('tag'); ap.add_argument('--var',default='A'); ap.add_argument('--obj',default='lambdarank')
ap.add_argument('--rel',type=int,default=40); ap.add_argument('--rounds',type=int,default=1500); ap.add_argument('--save',default='500,1000,1500'); ap.add_argument('--params',default='{}')
ap.add_argument('--threads',type=int,default=8); ap.add_argument('--seed',type=int,default=0)
a=ap.parse_args(); t0=time.time(); DR='^sq_(pdot|pcos|mdot|mcos)$'
extra=tuple(f'feat_lr2,feat_tt,feat_pp,feat_sqC,feat_dw,feat_{a.var}p,feat_N{a.var}s'.split(','))
trn=[f'f{k}' for k in range(max(1,a.cut-24),a.cut-20)]; vn='eval' if a.cut==27 else f'val{a.cut}'
gain=json.load(open(f'/workspace/huawei/v19/ens/nm{a.var}s_c{25 if a.cut!=23 else 23}_s0_gain.json'))
def relf(X,fl,top):
    ix=[fl.index(c) for c in top]; Z=X[:,ix].reshape(-1,10,len(ix)).astype(np.float32)
    with np.errstate(all='ignore'):
        mu=np.nanmean(Z,1,keepdims=True); mx=np.nanmax(Z,1,keepdims=True)
        rk=(np.argsort(np.argsort(np.where(np.isnan(Z),-np.inf,Z),1),1)).astype(np.float32)/9.
    return np.concatenate([(Z-mu).reshape(-1,len(ix)),(Z-mx).reshape(-1,len(ix)),rk.reshape(-1,len(ix))],1)
Xs=[];ys=[];fl=None
for n in trn:
    X,m,f=matrix(n,1.0,extra=extra); keep=[i for i,c in enumerate(f) if not re.search(DR,c)]
    if fl is None: fl=[f[i] for i in keep]; top=[c for c,_ in sorted(gain.items(),key=lambda kv:-kv[1]) if c in fl][:a.rel]
    X=np.ascontiguousarray(X[:,keep]); Xs.append(np.concatenate([X,relf(X,fl,top)],1) if a.rel>0 else X); ys.append(m['clicked'].to_numpy()); del X,m; gc.collect()
X=np.concatenate(Xs); del Xs; y=np.concatenate(ys); gc.collect(); E=len(y)//10; assert len(y)%10==0
names=fl+[f'{p}_{c}' for p in ('dmu','dmx','rk') for c in top] if a.rel>0 else fl
P=dict(objective=a.obj,learning_rate=0.02,num_leaves=127,min_data_in_leaf=200,feature_fraction=0.5,bagging_fraction=0.8,bagging_freq=1,lambda_l2=30.0,max_bin=255,force_col_wise=True,verbose=-1,num_threads=a.threads,seed=42+a.seed)
if a.obj=='lambdarank': P.update(lambdarank_truncation_level=10,eval_at=[10],label_gain=[0,1])
P.update(json.loads(a.params)); print('load',round(time.time()-t0),'s',X.shape,flush=True)
ds=lgb.Dataset(X,y,feature_name=names,free_raw_data=True,group=None if a.obj=='binary' else [10]*E,params=dict(max_bin=255,force_col_wise=True)).construct(); del X; gc.collect()
Xv,mv,fv=matrix(vn,1.0,extra=extra); Xv=Xv[:,[fv.index(c) for c in fl]]; Xv=np.concatenate([Xv,relf(Xv,fl,top)],1) if a.rel>0 else Xv
RS=[int(r) for r in a.save.split(',')]; t1=time.time(); mdl=lgb.train(P,ds,max(RS)); t_fit=time.time()-t1
S=np.stack([mdl.predict(Xv,num_iteration=r,raw_score=True) for r in RS],1).astype(np.float32); np.save(f'ens/{a.tag}_{a.var}_c{a.cut}_s{a.seed}.npy',S)
json.dump(dict(zip(names,mdl.feature_importance('gain').tolist())),open(f'ens/{a.tag}_{a.var}_c{a.cut}_s{a.seed}_gain.json','w'))
msg=dict(cut=a.cut,tag=a.tag,var=a.var,obj=a.obj,rel=a.rel,nf=len(names),fit_s=round(t_fit),min=round((time.time()-t0)/60,1),params=a.params)
if a.cut!=27:
    yv_=mv['clicked'].to_numpy(); msg['auc']={r:round(roc_auc_score(yv_,S[:,i]),5) for i,r in enumerate(RS)}
    msg['auc_within_slate_demeaned']={r:round(roc_auc_score(yv_,(S[:,i].reshape(-1,10)-S[:,i].reshape(-1,10).mean(1,keepdims=True)).reshape(-1)),5) for i,r in enumerate(RS)}
print(json.dumps(msg),flush=True); open('logs/gbdt_log.jsonl','a').write(json.dumps(msg)+'\n')
