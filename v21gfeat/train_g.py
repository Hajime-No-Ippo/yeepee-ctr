"""train_g.py CUT VARIANT [--seed S --rounds N --threads 2 --save 300,600,900]  VARIANT: num (top-80 numeric control, same as v21cat 'num') | numg (top-80 + graph feats)
Same LightGBM params/rel-feature recipe as v21cat/train.py. Train tables f{max(1,cut-24)}..f{cut-21}; graph feats for f_j come from graph k=j (before the rows' own day). cut 27 = eval (train f3..f6, graph k=7 for eval)."""
import os,sys,re,json,gc,time,argparse; sys.dont_write_bytecode=True
sys.path.insert(0,'/workspace/huawei/v19'); os.chdir('/workspace/huawei/v21gfeat')
import numpy as np, lightgbm as lgb
ap=argparse.ArgumentParser(); ap.add_argument('cut',type=int); ap.add_argument('var'); ap.add_argument('--seed',type=int,default=0); ap.add_argument('--rounds',type=int,default=900)
ap.add_argument('--save',default='300,600,900'); ap.add_argument('--nnum',type=int,default=80); ap.add_argument('--threads',type=int,default=2); ap.add_argument('--params',default='{}'); ap.add_argument('--tag',default=''); ap.add_argument('--nrel',type=int,default=25)
a=ap.parse_args(); t0=time.time(); DR='^sq_(pdot|pcos|mdot|mcos)$'
trn=[f'f{k}' for k in range(max(1,a.cut-24),a.cut-20)]; vn='eval' if a.cut==27 else f'val{a.cut}'
useG=a.var=='numg'; gnames=json.load(open('cache/gf_names.json')) if useG else []
from cl import matrix
gain=json.load(open('/workspace/huawei/v19/ens/nmAs_c25_s0_gain.json')); extra=tuple('feat_lr2,feat_tt,feat_pp,feat_sqC,feat_dw,feat_Ap,feat_NAs'.split(','))
numcols=None; GT=os.environ.get('GTAG','')
def load(T):
    global numcols
    X,m,f=matrix(T,1.0,extra=extra); keep=[i for i,c in enumerate(f) if not re.search(DR,c)]; fl=[f[i] for i in keep]
    if numcols is None: numcols=[c for c,_ in sorted(gain.items(),key=lambda kv:-kv[1]) if c in fl][:a.nnum]
    X=X[:,[f.index(c) for c in numcols]]; parts=[X]; rid=m['rowid'].astype(str) if hasattr(m['rowid'],'astype') else np.asarray(m['rowid']).astype(str)
    if useG:
        assert (np.load(f'cache/gf{GT}_{T}_rowid.npy',allow_pickle=True).astype(str)==np.asarray(rid).astype(str)).all(),T   # graph-feature row order == numeric matrix row order
        parts.append(np.load(f'cache/gf{GT}_{T}.npy'))
    return np.concatenate(parts,1).astype(np.float32)
def relf(X,ix):
    Z=X[:,ix].reshape(-1,10,len(ix))
    with np.errstate(all='ignore'):
        mu=np.nanmean(Z,1,keepdims=True); mx=np.nanmax(Z,1,keepdims=True); rk=(np.argsort(np.argsort(np.where(np.isnan(Z),-np.inf,Z),1),1)).astype(np.float32)/9.
    return np.concatenate([(Z-mu).reshape(-1,len(ix)),(Z-mx).reshape(-1,len(ix)),rk.reshape(-1,len(ix))],1)
import polars as pl
def labels(T): return pl.read_parquet(f'/workspace/huawei/v9/feat_v3/{T}.parquet',columns=['clicked'])['clicked'].to_numpy()
Xs=[];ys=[]
for T in trn:
    X=load(T); Xs.append(X); ys.append(labels(T)); del X; gc.collect()
names=numcols+gnames
relcand=list(range(min(a.nrel,len(numcols))))+([len(numcols)+gnames.index(n) for n in ('g_suv','g_sh','g_cosm','g_cosx','g_G')] if useG else [])
X=np.concatenate(Xs); del Xs; y=np.concatenate(ys)
X=np.concatenate([X,relf(X,relcand)],1); allnames=names+[f'{p}_{names[i]}' for p in('dmu','dmx','rk') for i in relcand]
P=dict(objective='binary',learning_rate=0.03,num_leaves=63,min_data_in_leaf=300,feature_fraction=0.5,bagging_fraction=0.8,bagging_freq=1,lambda_l2=30.0,max_bin=255,force_col_wise=True,verbose=-1,num_threads=a.threads,seed=42+a.seed)
P.update(json.loads(a.params)); print('load',round(time.time()-t0),'s',X.shape,flush=True)
ds=lgb.Dataset(X,y,feature_name=allnames,free_raw_data=True,params=dict(max_bin=255,force_col_wise=True)).construct(); del X; gc.collect()
Xv=load(vn); Xv=np.concatenate([Xv,relf(Xv,relcand)],1)
RS=[int(r) for r in a.save.split(',')]; t1=time.time(); mdl=lgb.train(P,ds,max(RS))
S=np.stack([mdl.predict(Xv,num_iteration=r,raw_score=True) for r in RS],1).astype(np.float32)
tag=f'gf_{a.var}{a.tag}{GT}_c{a.cut}_s{a.seed}'; os.makedirs('ens',exist_ok=True); np.save(f'ens/{tag}.npy',S)
json.dump(dict(zip(allnames,mdl.feature_importance('gain').tolist())),open(f'ens/{tag}_gain.json','w'))
msg=dict(cut=a.cut,var=a.var,tag=a.tag,nf=len(allnames),seed=a.seed,fit_s=round(time.time()-t1),total_min=round((time.time()-t0)/60,1))
if a.cut!=27:
    from sklearn.metrics import roc_auc_score as AUC; yv=labels(vn); msg['auc']={r:round(AUC(yv,S[:,i]),5) for i,r in enumerate(RS)}
print(json.dumps(msg),flush=True); open('logs/train_log.jsonl','a').write(json.dumps(msg)+'\n')
