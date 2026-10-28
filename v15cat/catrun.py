"""CatBoost (CPU) with categorical sets. cwd=/workspace/huawei/v12. python catrun.py CUT TAG --set A|B [--iters N] [--negfrac .5] [--fixed_iters N]
Set A: cats + top-100 numeric v12 columns by lgb255 gain. Set B: all 435 v12 numeric + cats. Train folds f(max(1,cut-24))..f(cut-21) (same as dwbig); negatives subsampled to negfrac.
Early stopping on val{CUT} (as ordered by Eric) unless --fixed_iters (then no val used for stopping). Saves ens scores to v15cat/ens/{tag}_c{CUT}.npy"""
import sys,os,re,json,gc,time,argparse
sys.path.insert(0,'/workspace/huawei/v15cat/pylib'); sys.path.insert(0,'/workspace/huawei/v12x'); sys.dont_write_bytecode=True
import numpy as np, pandas as pd, polars as pl
from catboost import CatBoostClassifier, Pool
from cl import matrix
from sklearn.metrics import roc_auc_score
ap=argparse.ArgumentParser(); ap.add_argument('cut',type=int); ap.add_argument('tag'); ap.add_argument('--set',default='A'); ap.add_argument('--iters',type=int,default=3000)
ap.add_argument('--negfrac',type=float,default=.5); ap.add_argument('--fixed_iters',type=int,default=0); ap.add_argument('--depth',type=int,default=8); ap.add_argument('--lr',type=float,default=.05)
ap.add_argument('--threads',type=int,default=8); ap.add_argument('--topn',type=int,default=100); ap.add_argument('--dropcat',default='')
a=ap.parse_args(); t0=time.time(); DROP='^sq_(pdot|pcos|mdot|mcos)$'; extra=('feat_lr2','feat_tt','feat_pp','feat_sqC','feat_dw')
CATD=f'/workspace/huawei/v15cat/cat{a.cut}'; trn=[f'f{k}' for k in range(max(1,a.cut-24),a.cut-20)]; vn=f'val{a.cut}'
gain=json.load(open('/workspace/huawei/v12x/ens/lgb255_c25_s0_gain.json')); top=[c for c,_ in sorted(gain.items(),key=lambda x:-x[1])[:a.topn]]
def load(n,keep_neg=1.0,seed=0):
    X,m,f=matrix(n,1.0,variant='notime',extra=extra); keep=[i for i,c in enumerate(f) if not re.search(DROP,c)]; fl=[f[i] for i in keep]
    if a.set=='A': sel=[fl.index(c) for c in top]; fl=[fl[i] for i in sel]; keep=[keep[i] for i in sel]
    cat=pl.read_parquet(f'{CATD}/{n}.parquet'); assert (cat['rowid'].to_numpy()==m['rowid'].to_numpy()).all(),'row order mismatch'
    y=m['clicked'].to_numpy(); r=np.random.RandomState(seed+len(y)).rand(len(y)); ix=np.where((y==1)|(r<keep_neg))[0] if keep_neg<1 else np.arange(len(y))
    df=pd.DataFrame(np.ascontiguousarray(X[ix][:,keep]),columns=fl); cc=[c for c in cat.columns if c!='rowid' and not (a.dropcat and re.search(a.dropcat,c))]
    for c in cc: df[c]=cat[c].to_numpy()[ix]
    del X; gc.collect(); return df,y[ix],cc
parts=[load(n,a.negfrac) for n in trn]; cc=parts[0][2]; Xt=pd.concat([p[0] for p in parts],ignore_index=True); yt=np.concatenate([p[1] for p in parts]); del parts; gc.collect()
Xv,yv,_=load(vn,1.0); print('train',Xt.shape,'val',Xv.shape,'pos rate',yt.mean(),'cats',len(cc),flush=True); t_load=time.time()-t0
ptr=Pool(Xt,yt,cat_features=cc); pv=Pool(Xv,yv,cat_features=cc); del Xt; gc.collect()
P=dict(loss_function='Logloss',eval_metric='AUC',depth=a.depth,learning_rate=a.lr,l2_leaf_reg=10,iterations=a.fixed_iters or a.iters,thread_count=a.threads,random_seed=42,
       max_ctr_complexity=1,border_count=128,od_type='Iter',od_wait=100,metric_period=25,verbose=100,allow_writing_files=False)
if a.fixed_iters: P.pop('od_type'); P.pop('od_wait')
m=CatBoostClassifier(**P); t1=time.time(); m.fit(ptr,eval_set=pv if not a.fixed_iters else None,use_best_model=not a.fixed_iters); t_fit=time.time()-t1
s=m.predict(pv,prediction_type='RawFormulaVal'); os.makedirs('/workspace/huawei/v15cat/ens',exist_ok=True); np.save(f'/workspace/huawei/v15cat/ens/{a.tag}_c{a.cut}.npy',s.astype(np.float32))
imp=dict(zip(m.feature_names_,m.get_feature_importance().tolist())); json.dump(imp,open(f'/workspace/huawei/v15cat/ens/{a.tag}_c{a.cut}_imp.json','w'))
msg=dict(cut=a.cut,tag=a.tag,set=a.set,n=len(yt),nf=Xv.shape[1],ncat=len(cc),best_iter=int(m.get_best_iteration()) if not a.fixed_iters else a.fixed_iters,auc=round(roc_auc_score(yv,s),5),
         load_s=round(t_load),fit_s=round(t_fit),min=round((time.time()-t0)/60,1),es_on_val=not a.fixed_iters)
print(json.dumps(msg),flush=True); open('/workspace/huawei/v15cat/ens/clog.jsonl','a').write(json.dumps(msg)+'\n')
