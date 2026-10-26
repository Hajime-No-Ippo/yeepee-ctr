# DIAGNOSTIC ONLY: adversarial validation, train folds f3..f6 (c27 training rows) vs eval rows on the v7 feature matrix.
# No features are created from this; dropping features based on it is a proposal for Eric's approval.
import sys, json, numpy as np, lightgbm as lgb
sys.path.insert(0,'/workspace/huawei/v2work')
from cl import matrix
from sklearn.metrics import roc_auc_score as A
from sklearn.model_selection import StratifiedKFold
EX=('feat_lr2','feat_tt'); rs=np.random.RandomState(0); N=150000
Xs=[];f=None
for n in ('f3','f4','f5','f6'):
    X,m,f=matrix(n,1.0,extra=EX); Xs.append(X[rs.rand(len(X))<0.12])
Xt=np.concatenate(Xs); Xt=Xt[rs.choice(len(Xt),min(N,len(Xt)),replace=False)]
Xe,me,fe=matrix('eval',1.0,extra=EX); assert fe==f; Xe=Xe[rs.choice(len(Xe),N,replace=False)]
X=np.concatenate([Xt,Xe]); y=np.r_[np.zeros(len(Xt)),np.ones(len(Xe))]
P=dict(objective='binary',learning_rate=0.1,num_leaves=31,min_data_in_leaf=200,feature_fraction=0.8,verbose=-1,num_threads=4)
def cv(X,cols):
    oof=np.zeros(len(y)); imp=np.zeros(len(cols))
    for tr,va in StratifiedKFold(3,shuffle=True,random_state=0).split(X,y):
        b=lgb.train(P,lgb.Dataset(X[tr],y[tr],feature_name=cols),150); oof[va]=b.predict(X[va]); imp+=b.feature_importance('gain')
    return A(y,oof),imp/imp.sum()
auc,imp=cv(X,f); top=np.argsort(-imp)[:20]
single={f[i]:round(abs(A(y,np.nan_to_num(X[:,i],nan=-9e9))-0.5)+0.5,4) for i in top}
res=dict(adv_auc_all=round(auc,4),top_gain=[(f[i],round(float(imp[i]),4),single[f[i]]) for i in top])
# without the obviously time-dependent columns (counts that grow with the window)
keep=[i for i,c in enumerate(f) if i not in set(top[:10])]
auc2,_=cv(X[:,keep],[f[i] for i in keep]); res['adv_auc_without_top10']=round(auc2,4)
json.dump(res,open('/workspace/huawei/research/v7_err_scripts/adv.json','w'),indent=1); print(json.dumps(res,indent=0))
