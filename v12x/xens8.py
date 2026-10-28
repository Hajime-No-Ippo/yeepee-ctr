"""XGBoost on the v8-lite feature set (regenerated v8 work dir), c27 only, same folds/negfrac as v8 'slow' member. cwd = v8regen/work"""
import sys, re, json, gc, time, numpy as np
sys.path.insert(0,'/workspace/huawei/v12x/pylib'); sys.path.insert(0,'.')
import xgboost as xgb
from cl import matrix
t0=time.time(); W='/workspace/huawei/v12x/v8regen/work'; extra=('feat_lr2','feat_tt','feat_pp')
Xs=[];ys=[]
for n in ['f3','f4','f5','f6']:
    X,m,f=matrix(n,1.0,variant='notime',extra=extra); Xs.append(X); ys.append(m['clicked'].to_numpy()); del X,m; gc.collect()
X=np.concatenate(Xs); y=np.concatenate(ys); del Xs
rs=np.random.RandomState(123); sel=(y==1)|(rs.rand(len(y))<0.5); X=np.clip(np.ascontiguousarray(X[sel]),-1e15,1e15); y=y[sel]
P=dict(objective='binary:logistic',tree_method='hist',grow_policy='lossguide',max_leaves=127,max_depth=0,eta=0.03,subsample=0.8,colsample_bytree=0.5,min_child_weight=30,reg_lambda=30,max_bin=255,nthread=8,seed=42)
d=xgb.QuantileDMatrix(X,y,feature_names=f,max_bin=255); del X; gc.collect()
Xv,mv,fv=matrix('eval',1.0,variant='notime',extra=extra); Xv=np.clip(Xv[:,[fv.index(c) for c in f]],-1e15,1e15)
b=xgb.train(P,d,450); s=b.predict(xgb.DMatrix(Xv,feature_names=f),output_margin=True)
np.save('/workspace/huawei/v12x/v8regen/xgb8_eval_rowid.npy',np.array(mv['rowid'].to_list()).astype(str)); np.save('/workspace/huawei/v12x/v8regen/xgb8_eval.npy',s.astype(np.float32))
print('done',len(y),len(f),round(time.time()-t0)); 
