import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='2'
import numpy as np, polars as pl, json
from scipy.stats import rankdata
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score as AUC
H='/workspace/huawei'; np.random.seed(0)
names=['z','M','big','list1','D','ref']
D={}
for c in (23,25):
    z=np.load(f'{H}/cb_blend/oof_c{c}.npz',allow_pickle=True)
    ctx=pl.read_parquet(f'{H}/v12/feat_v3/val{c}.parquet',columns=['eid','u','us_nev','tm_stale','hist_n_all','us_ctr'])
    D[c]=dict(y=z['y'].astype(int),X=np.column_stack([z['raw_'+n].astype(np.float64) for n in names]),ctx=ctx)
    assert (z['rowid']==pl.read_parquet(f'{H}/v12/feat_v3/val{c}.parquet',columns=['rowid'])['rowid'].to_numpy()).all()
rk=lambda x:(rankdata(x)-1)/(len(x)-1)
sig=lambda x:1/(1+np.exp(-x))
def tr(c,kind):
    X=D[c]['X']
    if kind=='logit': return X
    if kind=='rank': return np.column_stack([rk(X[:,j]) for j in range(X.shape[1])])
    if kind=='prob': return sig(X)
    if kind=='zlogit': return (X-X.mean(0))/X.std(0)
res={}
def rep(k,v): res[k]=v; print(k,v,flush=True)
for c in (23,25):
    for j,n in enumerate(names): rep(f'single_{n}_c{c}',round(AUC(D[c]['y'],D[c]['X'][:,j]),5))
# 1) equal-weight combos under 4 transforms
sets={'zM':[0,1],'zMbig':[0,1,2],'zMbigD':[0,1,2,4],'all6':list(range(6)),'zMbigDlist':[0,1,2,3,4],'MbigDlist':[1,2,3,4],'zMD':[0,1,4]}
for kind in ('logit','rank','prob','zlogit'):
    for s,idx in sets.items():
        for c in (23,25):
            X=tr(c,kind)[:,idx]; rep(f'eq_{kind}_{s}_c{c}',round(AUC(D[c]['y'],X.mean(1)),5))
# 2) weight fitting cross-cutoff, logit space, Nelder-Mead softmax weights; Caruana
def fitw(Xs,ys,k):
    def f(th):
        w=np.exp(th-th.max());w/=w.sum(); return -np.mean([AUC(y,X@w) for X,y in zip(Xs,ys)])
    r=minimize(f,np.zeros(k),method='Nelder-Mead',options=dict(xatol=2e-3,fatol=1e-7,maxiter=150*k,adaptive=True))
    w=np.exp(r.x-r.x.max());return w/w.sum()
def caruana(X,y,iters=40,init=1):
    k=X.shape[1];cnt=np.zeros(k);S=np.zeros(len(y));n=0
    for i in range(iters):
        best=None
        for j in range(k):
            a=AUC(y,(S+X[:,j])/(n+1))
            if best is None or a>best[0]: best=(a,j)
        cnt[best[1]]+=1;S+=X[:,best[1]];n+=1
    return cnt/cnt.sum()
# subsample rows for speed: AUC on full, fit on full ok (500k) but NM costly; use user-hash half? use full.
for kind in ('logit','rank'):
    for s in ('all6',):
        idx=sets[s]
        for a,b in ((23,25),(25,23)):
            Xa=tr(a,kind)[:,idx]; Xb=tr(b,kind)[:,idx]
            w=fitw([Xa],[D[a]['y']],len(idx)); rep(f'NMw_{kind}_{s}_fit{a}_test{b}',dict(w=[round(float(x),3) for x in w],auc=round(AUC(D[b]['y'],Xb@w),5),eq=round(AUC(D[b]['y'],Xb.mean(1)),5)))
            w=caruana(Xa,D[a]['y'],30); rep(f'Caruana_{kind}_{s}_fit{a}_test{b}',dict(w=[round(float(x),3) for x in w],auc=round(AUC(D[b]['y'],Xb@w),5)))
# 3) stacking meta-models, fit on one cutoff, test other. features: member logits (+ event context)
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
def evfeats(c,s):
    ctx=D[c]['ctx'].with_columns(pl.Series('s',s))
    ctx=ctx.with_columns(pl.col('s').max().over('eid').alias('emax'),pl.col('s').mean().over('eid').alias('emean'),pl.col('s').std().over('eid').alias('estd'),pl.col('s').rank('ordinal',descending=True).over('eid').alias('erank'))
    ctx=ctx.with_columns((pl.col('s')-pl.col('emean')).alias('edev'),(pl.col('s')-pl.col('emax')).alias('egap'))
    return ctx
def meta_X(c,mode):
    X=D[c]['X'];base=X[:,[0,1,2,4]].mean(1)  # z,M,big,D equal logit
    cols=[X]
    if mode=='lr_members': return X
    ctx=evfeats(c,base)
    if mode=='gbm_ev' or mode=='lr_ev':
        cols.append(ctx.select(['emax','emean','estd','edev','egap']).to_numpy())
    if mode=='gbm_ev_user':
        cols.append(ctx.select(['emax','emean','estd','edev','egap','us_nev','tm_stale','hist_n_all','us_ctr']).to_numpy())
    if mode=='gbm_user_only':
        cols.append(ctx.select(['us_nev','tm_stale','hist_n_all','us_ctr']).to_numpy())
    if mode=='gbm_members':
        pass
    return np.column_stack(cols)
def gbm_fit(Xa,ya,seed=0,n=150):
    p=dict(objective='binary',learning_rate=0.05,num_leaves=7,min_data_in_leaf=2000,feature_fraction=0.8,bagging_fraction=0.7,bagging_freq=1,lambda_l2=50,num_threads=2,verbose=-1,seed=seed)
    return lgb.train(p,lgb.Dataset(Xa,ya),n)
for mode in ('lr_members','lr_ev','gbm_members','gbm_ev','gbm_user_only','gbm_ev_user'):
    for a,b in ((23,25),(25,23)):
        Xa=meta_X(a,mode);Xb=meta_X(b,mode)
        if mode.startswith('lr'):
            mu,sd=Xa.mean(0),Xa.std(0);m=LogisticRegression(C=0.01,max_iter=300).fit((Xa-mu)/sd,D[a]['y']);s=m.decision_function((Xb-mu)/sd)
        else:
            m=gbm_fit(Xa,D[a]['y']);s=m.predict(Xb,raw_score=True)
        base=D[b]['X'][:,[0,1,2,4]].mean(1)
        rep(f'meta_{mode}_fit{a}_test{b}',dict(auc=round(AUC(D[b]['y'],s),5),base_zMbigD_eq=round(AUC(D[b]['y'],base),5),delta=round(AUC(D[b]['y'],s)-AUC(D[b]['y'],base),5)))
json.dump(res,open('screen_res.json','w'),indent=1)
