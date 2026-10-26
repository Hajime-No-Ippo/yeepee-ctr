import numpy as np, polars as pl, lightgbm as lgb, gc
from cl import matrix
from sklearn.metrics import roc_auc_score
Xa,ma,f=matrix('val25',0.25); Xb,mb,fb=matrix('eval',0.25); assert f==fb
X=np.vstack([Xa,Xb]); y=np.r_[np.zeros(len(Xa)),np.ones(len(Xb))]; g=np.r_[ma['eid'].to_numpy(),mb['eid'].to_numpy()]
rng=np.random.default_rng(0); ue=np.unique(g); fold=dict(zip(ue,rng.integers(0,4,len(ue)))); fo=np.array([fold[e] for e in g])
P=dict(objective='binary',learning_rate=0.1,num_leaves=31,min_data_in_leaf=200,feature_fraction=0.8,verbose=-1,num_threads=6)
def run(cols,tag):
    idx=[f.index(c) for c in cols]; oof=np.zeros(len(y)); imp=np.zeros(len(cols))
    for k in range(4):
        tr=fo!=k; m=lgb.train(P,lgb.Dataset(X[tr][:,idx],y[tr]),150); oof[~tr]=m.predict(X[~tr][:,idx]); imp+=m.feature_importance('gain')
    o=np.argsort(-imp)[:15]; print(tag,'adv AUC',round(roc_auc_score(y,oof),4),'top',[(cols[i],round(imp[i]/imp.sum(),3)) for i in o],flush=True)
    return [cols[i] for i in o]
top=run(f,'all-feats')
# drop the obvious time-structure drift and re-run, to see what else drifts
f2=[c for c in f if c not in top[:5]]; run(f2,'minus-top5')
itemish=lambda c: c.startswith(('it_','ev_it_','v2_i_','v2_ip','v2_ev_i','mf_b','v2_itb','v2_ev_ip')) or 'imp' in c
f3=[c for c in f if not itemish(c)]; print('n non-item feats',len(f3)); t3=run(f3,'non-item')
f4=[c for c in f3 if c not in t3[:5]]; run(f4,'non-item-minus-top5')
