"""Compare v2 (my features) vs v3 features vs blend on the shared validation:
cutoff Jan25, targets = first event per user in [Jan25, Jan27); training rows = first-event folds with cutoffs Jan21..Jan24 (W=1d)."""
import sys, os, numpy as np, polars as pl, lightgbm as lgb, json
sys.path.insert(0,'/workspace/huawei/v8'); sys.path.insert(0,'/workspace/huawei/v3')
from feats import DROP
from features import feature_cols, GROUPS
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
P2='/workspace/huawei/v8/feat'; P3='/workspace/huawei/v3/feat'
def fit_pred(tr, va, feats, params, rounds, ev_every=25):
    Xtr=tr.select(feats).to_numpy().astype(np.float32); Xva=va.select(feats).to_numpy().astype(np.float32); y=va['clicked'].to_numpy()
    m=lgb.train(params,lgb.Dataset(Xtr,tr['clicked'].to_numpy(),feature_name=feats),num_boost_round=rounds)
    curve=[(k,roc_auc_score(y,m.predict(Xva,num_iteration=k))) for k in range(ev_every,rounds+1,ev_every)]
    return m,curve,Xva
base=dict(objective='binary',num_leaves=63,min_data_in_leaf=200,feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10.0,verbose=-1,seed=42,num_threads=8)
# ---- v3 features
tr3=pl.concat([pl.read_parquet(f'{P3}/f{k}.parquet') for k in (1,2,3,4)],how='diagonal_relaxed')
va3=pl.read_parquet(f'{P3}/val25.parquet')
f3=feature_cols(tr3,GROUPS)
m3,c3,X3=fit_pred(tr3,va3,f3,dict(base,learning_rate=0.05),500)
print('v3 curve',[(k,round(a,5)) for k,a in c3],flush=True)
b3=max(c3,key=lambda x:x[1]); p3=m3.predict(X3,num_iteration=b3[0])
# ---- v2 features
tr2=pl.concat([pl.read_parquet(f'{P2}/h{k}.parquet') for k in (1,2,3,4)],how='diagonal_relaxed')
va2=pl.read_parquet(f'{P2}/val25.parquet')
f2=[c for c in tr2.columns if c not in DROP]
m2,c2,X2=fit_pred(tr2,va2,f2,dict(base,learning_rate=0.05),500)
print('v2 curve',[(k,round(a,5)) for k,a in c2],flush=True)
b2=max(c2,key=lambda x:x[1]); p2=m2.predict(X2,num_iteration=b2[0])
# ---- combined feature set (v3 + v2 features not in v3), joined on rowid
extra=va2.select(['rowid']+[c for c in f2]).rename({c:'v2_'+c for c in f2})
trc=tr3.join(tr2.select(['rowid']+f2).rename({c:'v2_'+c for c in f2}),on='rowid',how='left')
vac=va3.join(extra,on='rowid',how='left')
fc=f3+['v2_'+c for c in f2]
mc,cc,Xc=fit_pred(trc,vac,fc,dict(base,learning_rate=0.05),500)
print('combined curve',[(k,round(a,5)) for k,a in cc],flush=True)
# ---- blends on rowid
a=pl.DataFrame({'rowid':va3['rowid'],'p3':p3,'y':va3['clicked']}).join(pl.DataFrame({'rowid':va2['rowid'],'p2':p2}),on='rowid')
y=a['y'].to_numpy(); r3=rankdata(a['p3'])/len(a); r2=rankdata(a['p2'])/len(a)
res=dict(v3=b3,v2=b2,combined=max(cc,key=lambda x:x[1]))
for w in (0.5,0.6,0.7,0.8,0.9): res[f'rankblend_w3={w}']=roc_auc_score(y,w*r3+(1-w)*r2)
print(json.dumps(res,indent=1))
imp=sorted(zip(fc,mc.feature_importance('gain')),key=lambda x:-x[1]); tot=sum(v for _,v in imp)
print('combined top', [(f,round(v/tot,4)) for f,v in imp[:25]])
