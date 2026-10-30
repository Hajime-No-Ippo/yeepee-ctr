import os,sys,re,json,gc,time
sys.dont_write_bytecode=True; sys.path.insert(0,'/workspace/huawei/v19')
import numpy as np, lightgbm as lgb
from cl import matrix
from sklearn.metrics import roc_auc_score
OUT='/workspace/huawei/dw_featcut/null'
EXTRA=('feat_lr2','feat_tt','feat_pp','feat_sqC','feat_dw','feat_Ap','feat_NAs')   # variant A, 484 cols (+4 raw sq_ cols removed by DROP)
DROP='^sq_(pdot|pcos|mdot|mcos)$'
S_RE=open('/workspace/huawei/v19cut/drop_S_A.txt').read().strip()
P=dict(objective='binary',learning_rate=0.015,num_leaves=127,min_data_in_leaf=200,feature_fraction=0.5,bagging_fraction=0.8,bagging_freq=1,lambda_l2=30.0,max_bin=255,force_col_wise=True,verbose=-1)
def trn(cut): return [f'f{k}' for k in range(max(1,cut-24),cut-20)]
def load(cut, drop_re):
    """train matrix (S-dropped by default), labels, val matrix/labels. Rows event-major (10/event) - verified via eid."""
    Xs=[];ys=[]; ok=True
    for n in trn(cut):
        X,m,f=matrix(n,1.0,variant='notime',extra=EXTRA); keep=[i for i,c in enumerate(f) if not re.search(DROP,c) and not re.search(drop_re,c)]; fl=[f[i] for i in keep]
        e=m['eid'].to_numpy(); ok&=bool((e.reshape(-1,10)==e.reshape(-1,10)[:,:1]).all())
        Xs.append(np.ascontiguousarray(X[:,keep])); ys.append(m['clicked'].to_numpy()); del X,m; gc.collect()
    assert ok, 'rows not event-major blocks of 10'
    X=np.concatenate(Xs); del Xs; y=np.concatenate(ys); gc.collect(); assert len(y)%10==0
    Xv,mv,fv=matrix(f'val{cut}',1.0,variant='notime',extra=EXTRA); Xv=np.ascontiguousarray(Xv[:,[fv.index(c) for c in fl]]); yv=mv['clicked'].to_numpy(); del mv
    return X,y,Xv,yv,fl
def log(path,o): open(path,'a').write(json.dumps(o)+'\n')
