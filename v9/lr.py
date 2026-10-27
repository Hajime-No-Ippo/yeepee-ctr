"""plain L2 logistic regression member. python lr.py CUT [C]"""
import sys, json, gc, time, numpy as np, polars as pl
from cl import matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score as A
cut=int(sys.argv[1]); Creg=float(sys.argv[2]) if len(sys.argv)>2 else 0.1; t0=time.time(); EX=('feat_lr2','feat_tt','feat_pp')
trn=[f'f{k}' for k in range(max(1,cut-24),cut-20)]
lens=[len(pl.read_parquet(f'feat_v3/{n}.parquet',columns=['rowid'])) for n in trn]
R=np.random.RandomState(123).rand(sum(lens)); Xs=[];ys=[];off=0
for n,L in zip(trn,lens):
    X,m,f=matrix(n,1.0,extra=EX); yy=m['clicked'].to_numpy(); sel=(yy==1)|(R[off:off+L]<0.5); off+=L; Xs.append(X[sel]); ys.append(yy[sel]); del X,m; gc.collect()
X=np.concatenate(Xs); del Xs; y=np.concatenate(ys); gc.collect()
X[~np.isfinite(X)]=np.nan
lo=np.nanpercentile(X[::7],0.5,axis=0); hi=np.nanpercentile(X[::7],99.5,axis=0); med=np.nanmedian(X[::7],axis=0)
def prep(X):
    nanm=np.isnan(X); X=np.where(nanm,med,X); np.clip(X,lo,hi,out=X); return X
X=prep(X); mu=X.mean(0); sd=X.std(0)+1e-6; X-=mu; X/=sd
# add nan indicators for columns with any nan? keep plain
mdl=LogisticRegression(C=Creg,max_iter=300,tol=1e-4); mdl.fit(X,y); del X; gc.collect()
vn='eval' if cut==27 else f'val{cut}'; Xv,mv,_=matrix(vn,1.0,extra=EX); Xv[~np.isfinite(Xv)]=np.nan; Xv=(prep(Xv)-mu)/sd
s=mdl.decision_function(Xv).astype(np.float32); np.save(f'ens/lr_c{cut}.npy',s)
r=dict(cut=cut,C=Creg,n=len(y),s=round(time.time()-t0))
if cut!=27: r['auc']=round(A(mv['clicked'].to_numpy(),s),5)
print(json.dumps(r),flush=True); open('cmp_results.jsonl','a').write(json.dumps(dict(r,tag='lr'))+'\n')
