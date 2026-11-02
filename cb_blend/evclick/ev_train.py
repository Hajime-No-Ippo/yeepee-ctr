import sys; sys.path.insert(0,'.')
import numpy as np, polars as pl, lightgbm as lgb, time, json
from sklearn.metrics import roc_auc_score
TR={23:['f1','f2'],25:['f1','f2','f3','f4'],26:['f2','f3','f4','f5'],27:['f3','f4','f5','f6']}
def load(ns): return pl.concat([pl.read_parquet(f'ev_{n}.parquet') for n in ns])
drop={'eid','u','ts','rowid0','k','n'}
P=dict(objective='binary',learning_rate=.03,num_leaves=31,min_data_in_leaf=100,feature_fraction=.5,bagging_fraction=.8,bagging_freq=1,lambda_l2=10,num_threads=4,verbose=-1,seed=0)
NR=600
out={}
for cut in (23,25,26,27):
    tr=load(TR[cut]); feats=[c for c in tr.columns if c not in drop]
    Xtr=tr.select(feats).to_numpy().astype(np.float32); ytr=(tr['k'].to_numpy()>0).astype(int)
    vn='eval' if cut==27 else f'val{cut}'; va=pl.read_parquet(f'ev_{vn}.parquet'); Xva=va.select(feats).to_numpy().astype(np.float32)
    t0=time.time(); m=lgb.train(P,lgb.Dataset(Xtr,ytr),NR)
    res={}
    for it in (150,300,450,600):
        p=m.predict(Xva,num_iteration=it); 
        if cut!=27: res[it]=roc_auc_score((va['k'].to_numpy()>0).astype(int),p)
    p=m.predict(Xva,num_iteration=NR); out[cut]=dict(eid=va['eid'].to_numpy(),p=p)
    np.savez(f'evpred_c{cut}.npz',eid=va['eid'].to_numpy(),p=p,rowid0=va['rowid0'].to_numpy())
    imp=sorted(zip(m.feature_importance('gain'),feats),reverse=True)[:12]
    print(cut,len(ytr),'pos rate',ytr.mean().round(3),'event AUC by iters',{k:round(v,4) for k,v in res.items()},round(time.time()-t0),'s',flush=True)
    print('   top gain',[(f,int(g)) for g,f in imp[:8]],flush=True)
