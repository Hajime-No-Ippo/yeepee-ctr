"""Step C screening: python screen.py CUT TAG [--frac 0.25] [--seeds 1] [--rounds 400] [--ev] [--drop regex] [--folds f|e]
Trains on a user-hash sample of the training folds (cutoffs < CUT), evaluates on full val{CUT}. Uses cl.matrix cache."""
import sys, re, json, gc, argparse, numpy as np, polars as pl, lightgbm as lgb
from cl import matrix
from sklearn.metrics import roc_auc_score
ap=argparse.ArgumentParser(); ap.add_argument('cut',type=int); ap.add_argument('tag'); ap.add_argument('--frac',type=float,default=0.25)
ap.add_argument('--seeds',type=int,default=1); ap.add_argument('--rounds',type=int,default=400); ap.add_argument('--ev',action='store_true')
ap.add_argument('--drop',default=None); ap.add_argument('--threads',type=int,default=8); ap.add_argument('--obj',default='binary'); ap.add_argument('--folds',default='f'); ap.add_argument('--stale',action='store_true'); ap.add_argument('--rc',action='store_true'); ap.add_argument('--cv',action='store_true'); ap.add_argument('--mods',default=''); ap.add_argument('--maxfold',type=int,default=99); ap.add_argument('--seedoff',type=int,default=0); ap.add_argument('--lr',default='')
a=ap.parse_args()
extra=((f'feat_ev/c{a.cut}',) if a.ev else ())+(('feat_rc',) if a.rc else ())+(('feat_cv',) if a.cv else ())+tuple(x for x in a.lr.split(',') if x); var=('stale' if a.stale else 'notime')+''.join('+'+m for m in a.mods.split(',') if m)
efold={25:[f'e{k}' for k in range(1,5)],26:[f'd{k}' for k in range(1,6)]}[a.cut]
trn=([f'f{k}' for k in range(1,min(a.cut-20,a.maxfold+1))] if 'f' in a.folds else [])+(efold if 'e' in a.folds else [])
Xs=[];ys=[];gs=[];feats=None
for n in trn:
    X,m,f=matrix(n,a.frac,variant=var,extra=extra); feats=f; Xs.append(X); ys.append(m['clicked'].to_numpy()); gs.append(m['eid'].to_numpy()); del X; gc.collect()
X=np.concatenate(Xs); del Xs; y=np.concatenate(ys); g=np.concatenate(gs)
keep=[i for i,c in enumerate(feats) if not (a.drop and re.search(a.drop,c))]; fl=[feats[i] for i in keep]
if len(keep)<len(feats): X=X[:,keep]
P=dict(objective=a.obj,learning_rate=0.05,num_leaves=63,min_data_in_leaf=200 if a.frac>=0.5 else 100,feature_fraction=0.7,bagging_fraction=0.8,
       bagging_freq=1,lambda_l2=10.0,verbose=-1,num_threads=a.threads)
if a.obj=='lambdarank':
    o=np.argsort(g,kind='stable'); X=X[o]; y=y[o]; g=g[o]; _,grp=np.unique(g,return_counts=True)
    ds=lgb.Dataset(X,y,group=grp,feature_name=fl,free_raw_data=True).construct(); P.update(lambdarank_truncation_level=10,eval_at=[10])
else: ds=lgb.Dataset(X,y,feature_name=fl,free_raw_data=True).construct()
del X; gc.collect()
Xv,mv,fv=matrix(f'val{a.cut}',1.0,variant=var,extra=extra); Xv=Xv[:,[fv.index(c) for c in fl]]; yv=mv['clicked'].to_numpy()
pv=np.zeros(len(yv)); curve={}
for s in range(a.seeds):
    mdl=lgb.train(dict(P,seed=42+s+a.seedoff),ds,a.rounds)
    for k in range(100,a.rounds+1,100): curve[k]=curve.get(k,0)+mdl.predict(Xv,num_iteration=k,raw_score=True)/a.seeds
p=curve[a.rounds]
def evaucs(score):
    df=pl.DataFrame({'e':mv['eid'],'y':yv,'s':score})
    E=df.group_by('e').agg(pl.col('y').max(),pl.col('s').max())
    W=df.group_by('e').agg((pl.col('y').max()>pl.col('y').min()).alias('mx'),pl.col('y'),pl.col('s')).filter('mx')
    wa=np.mean([roc_auc_score(np.array(r[0]),np.array(r[1])) for r in W.select('y','s').iter_rows()])
    return round(roc_auc_score(E['y'],E['s']),5), round(wa,5)
cold=mv['us_cold'].to_numpy()==1; noh=mv['us_nohist'].to_numpy()==1; st=mv['tm_stale'].to_numpy()>=86400
seg={k:round(roc_auc_score(yv[s],p[s]),5) if s.sum()>0 else None for k,s in {'warm':~cold,'hist_only':cold&~noh,'empty':noh,'stale_ge1d':st,'stale_lt1d':~st}.items()}
ea,wa=evaucs(p)
res=dict(cut=a.cut,tag=a.tag,lr=a.lr,drop=a.drop,mods=a.mods,maxfold=a.maxfold,folds=a.folds,stale=a.stale,rc=a.rc,cv=a.cv,frac=a.frac,seeds=a.seeds,obj=a.obj,n_feats=len(fl),curve={k:round(roc_auc_score(yv,v),5) for k,v in curve.items()},
         event_auc=ea,within_auc=wa,seg=seg)
gain=mdl.feature_importance('gain'); res['top']=[(fl[i],round(float(gain[i]/gain.sum()),3)) for i in np.argsort(-gain)[:8]]
print(json.dumps(res),flush=True); open('/workspace/huawei/v8/screen_results.jsonl','a').write(json.dumps(res)+'\n')
np.save(f'/workspace/huawei/v8/scr_{a.tag}_c{a.cut}.npy',p)
