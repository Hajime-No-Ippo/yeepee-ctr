import os; os.environ.update(OMP_NUM_THREADS='2')
import sys; sys.dont_write_bytecode=True; sys.path.insert(0,'/workspace/huawei/v20')
import json,re,time,gc,numpy as np,pandas as pd,lightgbm as lgb
from cl import matrix
from sklearn.metrics import roc_auc_score
EXTRA=('feat_lr2','feat_tt','feat_pp','feat_sqC','feat_dw','feat_Ap','feat_NAs'); DROP='^sq_(pdot|pcos|mdot|mcos)$'
N=pd.read_csv('gain_table.csv',index_col=0); cut=N.index[N.tier.isin(['safe','probably_safe'])].tolist(); cutS=N.index[N.tier=='safe'].tolist()
rs=np.random.RandomState(1); Xs=[];ys=[]
for n,k in [('f3',100000),('f4',100000)]:
    X,m,f=matrix(n,1.0,variant='notime',extra=EXTRA); keep=[i for i,c in enumerate(f) if not re.search(DROP,c)]; fl=[f[i] for i in keep]
    s=np.sort(rs.choice(len(X),k,replace=False)); Xs.append(X[s][:,keep]); ys.append(m['clicked'].to_numpy()[s]); del X,m; gc.collect()
Xt=np.concatenate(Xs); yt=np.concatenate(ys); del Xs
X,m,f=matrix('val25',1.0,variant='notime',extra=EXTRA); s=np.sort(rs.choice(len(X),400000,replace=False)); Xv=X[s][:,[f.index(c) for c in fl]]; yv=m['clicked'].to_numpy()[s]; del X,m
print(Xt.shape,Xv.shape,flush=True)
P=dict(objective='binary',learning_rate=0.05,num_leaves=127,min_data_in_leaf=200,feature_fraction=0.5,bagging_fraction=0.8,bagging_freq=1,lambda_l2=30.0,max_bin=255,force_col_wise=True,verbose=-1,num_threads=2)
res=[]
for name,drop in [('full',[]),('cut_safe',cutS),('cut_safe+probably',cut)]:
    ix=[i for i,c in enumerate(fl) if c not in set(drop)]
    for seed in (0,1):
        t=time.time(); ds=lgb.Dataset(Xt[:,ix],yt,params=dict(max_bin=255,force_col_wise=True),free_raw_data=False); b=lgb.train({**P,'seed':seed},ds,300)
        a={r:round(roc_auc_score(yv,b.predict(Xv[:,ix],num_iteration=r,raw_score=True)),5) for r in (150,300)}
        r=dict(set=name,nf=len(ix),seed=seed,auc=a,fit_s=round(time.time()-t)); print(r,flush=True); res.append(r); json.dump(res,open('check_results.json','w'),indent=1)
