import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='2'
import numpy as np, polars as pl, lightgbm as lgb, json, sys
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score as AUC
H='/workspace/huawei'
def rk(x): return (rankdata(x)-1)/(len(x)-1)
MEM=['z','M','big','list1','D','ref','A']
D={}
for c in (23,25):
    z=np.load(f'{H}/cb_blend/oof_c{c}.npz',allow_pickle=True)
    v=pl.read_parquet(f'{H}/v12/feat_v3/val{c}.parquet',columns=['rowid','eid','u','clicked','hist_n_all','tm_stale','us_since_ev','us_nohist','us_nev','us_nclk_all','it_imp','it_ctr'])
    assert (v['rowid'].to_numpy()==z['rowid']).all()
    d=dict(y=v['clicked'].to_numpy(),u=v['u'].to_numpy(),eid=v['eid'].to_numpy(),meta=v.select(['hist_n_all','tm_stale','us_since_ev','us_nohist','us_nev','us_nclk_all','it_imp','it_ctr']).to_numpy().astype(float),raw={m:z[f'raw_{m}'].astype(float) for m in MEM if f'raw_{m}' in z.files})
    D[c]=d; print(c,len(d['y']),list(d['raw']))
common=[m for m in MEM if all(m in D[c]['raw'] for c in D)]
print('common',common)
def evfeat(d,names):
    # within-event features: member logit minus event mean, within-event rank
    eid=np.unique(d['eid'],return_inverse=True)[1]; cnt=np.bincount(eid)
    F=[];nm=[]
    for m in names:
        x=d['raw'][m]; mu=np.bincount(eid,x)/cnt; F+= [x,x-mu[eid[:]]]; nm+=[m,m+'_dev']
    return np.column_stack(F),nm
def feats(d,names,meta=True,evt=True):
    X=[np.column_stack([d['raw'][m] for m in names])]
    if evt: X.append(evfeat(d,names)[0][:,1::2])
    if meta: X.append(d['meta'])
    return np.column_stack(X)
P=dict(objective='binary',learning_rate=0.05,num_leaves=15,min_data_in_leaf=500,feature_fraction=0.8,bagging_fraction=0.7,bagging_freq=1,lambda_l2=10,num_threads=2,verbose=-1,seed=1)
def lgbfit(X,y,rounds):
    return lgb.train(P,lgb.Dataset(X,y),rounds)
res={}
base={c:AUC(D[c]['y'],.4*rk(D[c]['raw']['z'])+.6*rk(D[c]['raw']['M'])) for c in D}
print('base .4z+.6M',base)
# plain: equal mean-of-ranks of all common
for c in D: print('equal-rank all',c,AUC(D[c]['y'],sum(rk(D[c]['raw'][m]) for m in common)))
cfgs={'logits only':dict(meta=False,evt=False),'+event-dev':dict(meta=False,evt=True),'+meta':dict(meta=True,evt=False),'+meta+event-dev':dict(meta=True,evt=True)}
for nm,cf in cfgs.items():
    for tr,te in ((23,25),(25,23)):
        Xtr=feats(D[tr],common,**cf);Xte=feats(D[te],common,**cf)
        # rounds chosen by inner user-grouped holdout in train fold
        g=(pl.Series(D[tr]['u']).hash(seed=5).to_numpy()%5)
        va=g==0
        m=lgb.train(P,lgb.Dataset(Xtr[~va],D[tr]['y'][~va]),600,valid_sets=[lgb.Dataset(Xtr[va],D[tr]['y'][va])],callbacks=[lgb.early_stopping(50,verbose=False)])
        nr=m.best_iteration; m=lgbfit(Xtr,D[tr]['y'],max(nr,20))
        a=AUC(D[te]['y'],m.predict(Xte)); print(nm,f'train C{tr} -> test C{te}',round(a,5),'vs base',round(base[te],5),'delta',round(a-base[te],5),'rounds',nr,flush=True)
        res[f'{nm} {tr}->{te}']=dict(auc=a,base=base[te],delta=a-base[te],rounds=int(nr))
    # also in-fold user-grouped 5-fold CV
    for c in D:
        X=feats(D[c],common,**cf); g=(pl.Series(D[c]['u']).hash(seed=7).to_numpy()%5); oof=np.zeros(len(X))
        for k in range(5):
            m=lgbfit(X[g!=k],D[c]['y'][g!=k],150); oof[g==k]=m.predict(X[g==k])
        a=AUC(D[c]['y'],oof); print(nm,f'userCV C{c}',round(a,5),'delta',round(a-base[c],5),flush=True); res[f'{nm} cv{c}']=dict(auc=a,delta=a-base[c])
json.dump(res,open('stack_results.json','w'),indent=1)
