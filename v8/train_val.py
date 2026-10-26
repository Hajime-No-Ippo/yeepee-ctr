import polars as pl, numpy as np, lightgbm as lgb, sys
sys.path.insert(0,'/workspace/huawei/v8')
from feats import DROP
from sklearn.metrics import roc_auc_score
D='/workspace/huawei/v8/feat'
tr_names=sys.argv[1].split(',') if len(sys.argv)>1 else ['f1','f2','f3','f4']
obj=sys.argv[2] if len(sys.argv)>2 else 'binary'
tr=pl.concat([pl.read_parquet(f'{D}/{n}.parquet').with_columns(pl.lit(i).alias('fold')) for i,n in enumerate(tr_names)],how='diagonal_relaxed').sort('fold','eid','r')
import os
VAL=os.environ.get('VAL','val')
va=pl.read_parquet(f'{D}/{VAL}.parquet').sort('eid','r')
feats=[c for c in tr.columns if c not in DROP]
excl=[x for x in (sys.argv[3].split(',') if len(sys.argv)>3 and sys.argv[3] else [])]
feats=[f for f in feats if not any(f.startswith(e) for e in excl)]
Xtr=tr.select(feats).to_numpy().astype(np.float32); ytr=tr['clicked'].to_numpy()
Xva=va.select(feats).to_numpy().astype(np.float32); yva=va['clicked'].to_numpy()
LR=float(os.environ.get('LR','0.02'));NL=int(os.environ.get('NL','31'))
p=dict(objective=obj,learning_rate=LR,num_leaves=NL,min_data_in_leaf=500,feature_fraction=0.5,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10,verbose=-1,num_threads=8,metric='auc')
if obj=='lambdarank':
    p.update(metric='ndcg',eval_at=[10])
    gtr=tr.group_by(['fold','eid'],maintain_order=True).len()['len'].to_numpy(); gva=va.group_by('eid',maintain_order=True).len()['len'].to_numpy()
    dtr=lgb.Dataset(Xtr,ytr,group=gtr,feature_name=feats); dva=lgb.Dataset(Xva,yva,group=gva,reference=dtr)
else:
    dtr=lgb.Dataset(Xtr,ytr,feature_name=feats); dva=lgb.Dataset(Xva,yva,reference=dtr)
aucs=[]
def cb(env):
    if env.iteration%int(os.environ.get('EV','50'))==int(os.environ.get('EV','50'))-1:
        pr=env.model.predict(Xva,num_iteration=env.iteration+1); a=roc_auc_score(yva,pr); aucs.append((env.iteration+1,a)); print(env.iteration+1,round(a,5),flush=True)
m=lgb.train(p,dtr,num_boost_round=int(sys.argv[4]) if len(sys.argv)>4 else 1500,callbacks=[cb])
best=max(aucs,key=lambda x:x[1]); print('BEST',obj,tr_names,excl,best)
imp=sorted(zip(feats,m.feature_importance('gain')),key=lambda x:-x[1])
tot=sum(v for _,v in imp)
for f,v in imp[:30]: print(f'{f:22s} {v/tot:.4f}')
np.save(f'/workspace/huawei/v8/pred_{VAL}_{obj}.npy',m.predict(Xva,num_iteration=best[0]))
pb=m.predict(Xva,num_iteration=best[0])
def seg(name,mask):
    if mask.sum()>0 and 0<yva[mask].sum()<mask.sum(): print(f'  seg {name:14s} rows={mask.sum():7d} auc={roc_auc_score(yva[mask],pb[mask]):.4f}')
u_ev=va['u_ev'].to_numpy(); hl=va['h_len'].to_numpy(); dt=va['dt'].to_numpy()
seg('all',np.ones(len(yva),bool)); seg('warm',u_ev>0); seg('hist_only',(u_ev==0)&(hl>0)); seg('empty_hist',hl==0); seg('weekday',dt==0); seg('weekend',dt==1)
