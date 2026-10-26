"""Ranking member: lambdarank/xendcg grouped by eid on lite folds. python rank.py CUT TAG --lr ... [--obj lambdarank] [--rounds 400]
Saves ens/{TAG}_rk_c{CUT}.npy; reports auc and blend with ens/{TAG}_blend_c{CUT}.npy at weights."""
import json, gc, time, argparse, numpy as np, lightgbm as lgb
from cl import matrix
from sklearn.metrics import roc_auc_score
ap=argparse.ArgumentParser(); ap.add_argument('cut',type=int); ap.add_argument('tag'); ap.add_argument('--lr',default='')
ap.add_argument('--obj',default='lambdarank'); ap.add_argument('--rounds',type=int,default=400); ap.add_argument('--nfolds',type=int,default=4); ap.add_argument('--params',default='{}')
a=ap.parse_args(); t0=time.time(); extra=tuple(x for x in a.lr.split(',') if x)
trn=[f'f{k}' for k in range(max(1,a.cut-20-a.nfolds),a.cut-20)]
Xs=[];ys=[];gs=[];fl=None
for n in trn:
    X,m,f=matrix(n,1.0,extra=extra); fl=f; e=m['eid'].to_numpy(); y=m['clicked'].to_numpy()
    o=np.argsort(e,kind='stable'); e=e[o]; y=y[o]
    u,st,cnt=np.unique(e,return_index=True,return_counts=True); pos=np.add.reduceat(y,st)
    ok=np.repeat((pos>0)&(pos<cnt),cnt)  # groups with both classes only
    Xs.append(X[o][ok]); ys.append(y[ok]); gs.append(cnt[(pos>0)&(pos<cnt)]); del X,m; gc.collect()
X=np.concatenate(Xs); y=np.concatenate(ys); g=np.concatenate(gs); del Xs; gc.collect()
P=dict(objective=a.obj,learning_rate=0.05,num_leaves=63,min_data_in_leaf=200,feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,
       lambda_l2=10.0,verbose=-1,num_threads=4,max_bin=63,force_col_wise=True,seed=42); P.update(json.loads(a.params))
ds=lgb.Dataset(X,y,group=g,feature_name=fl,params=dict(max_bin=63)).construct(); del X; gc.collect(); t1=time.time()
mdl=lgb.train(P,ds,a.rounds); tf=time.time()-t1
vn='eval' if a.cut==27 else f'val{a.cut}'
Xv,mv,fv=matrix(vn,1.0,extra=extra); Xv=Xv[:,[fv.index(c) for c in fl]]
s=mdl.predict(Xv,raw_score=True).astype(np.float32); np.save(f'ens/{a.tag}_rk_c{a.cut}.npy',s)
r=dict(cut=a.cut,tag=a.tag,obj=a.obj,n=int(len(y)),ng=int(len(g)),fit_s=round(tf),total_s=round(time.time()-t0))
if a.cut!=27:
    yv=mv['clicked'].to_numpy(); r['auc_rk']=round(roc_auc_score(yv,s),5)
    z=np.load(f'ens/{a.tag}_blend_c{a.cut}.npy'); r['blend']=round(roc_auc_score(yv,z),5)
    sz=(s-s.mean())/s.std()*z.std()+z.mean()
    for w in (0.1,0.2,0.3,0.4): r[f'w{w}']=round(roc_auc_score(yv,(1-w)*z+w*sz),5)
print(json.dumps(r),flush=True); open('scr_results.jsonl','a').write(json.dumps(r)+'\n')
