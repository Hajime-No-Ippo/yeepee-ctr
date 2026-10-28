"""exp2: row-level residual boost on v12 logits with event-aggregate features (own slate only). OOF 5-fold by user."""
import sys, json, time, numpy as np, polars as pl, lightgbm as lgb, warnings; warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score as A
from sklearn.model_selection import GroupKFold
from cl import matrix, BASE
import resid as R
from z12 import z12
cut=int(sys.argv[1]); seed=int(sys.argv[2]) if len(sys.argv)>2 else 0; t0=time.time()
X,m,fl=matrix(f'val{cut}',1.0,extra=BASE+('feat_sqC','feat_dw')); y=m['clicked'].to_numpy(); z=z12(cut)
user=pl.read_parquet(f'{R.F3}/val{cut}.parquet',columns=['u'])['u'].to_numpy()
n=len(y)//10; X3=X.reshape(n,10,-1)
const=(np.nanmax(X3,1)-np.nanmin(X3,1)==0).all(0)|np.isnan(X3[:,0,:]).all(0)
V=X3[:,:,~const]; vn=[c for c,k in zip(fl,~const) if k]
hl=R.hist_len(m['rowid'].to_numpy()); zz,dm,rk=R.wev(z); Z=z.reshape(-1,10)
Xb=np.column_stack([z,zz,dm,rk,np.repeat(np.log(np.exp(Z).sum(1)),10),np.repeat(Z.max(1),10),np.repeat(Z.mean(1),10),hl,(hl==0)]).astype(np.float32)
yc=(y.reshape(n,10)-y.reshape(n,10).mean(1,keepdims=True)).ravel()
Vd=np.nan_to_num(V-np.nanmean(V,1,keepdims=True)).reshape(n*10,-1).astype(np.float32)
sc=np.abs((Vd*yc[:,None]).sum(0))/(np.sqrt((Vd**2).sum(0))+1e-9); top=np.argsort(-sc)[:80]; print('top',[vn[i] for i in top[:15]],flush=True)
V=V[:,:,top]; RD=Vd[:,top]; del Vd
C=np.repeat(X3[:,0,const],10,0)
agg=lambda f:np.repeat(f(V,1),10,0)
EA=np.hstack([C,agg(np.nanmean),agg(np.nanmax),agg(np.nanmin),agg(np.nanstd)]).astype(np.float32)
sets={'base':Xb,'+evconst+agg':np.hstack([Xb,EA]),'+rowdev':np.hstack([Xb,RD]),'+all':np.hstack([Xb,EA,RD])}
print('built',{k:v.shape for k,v in sets.items()},time.time()-t0,flush=True)
res={}
for name,F in sets.items():
    p=np.zeros(len(y))
    for tr,te in GroupKFold(5).split(F,y,user): p[te]=R.fit_pred(F[tr],y[tr],z[tr],F[te],z[te],seed)
    res[name]=A(y,p); r=dict(cut=cut,seed=seed,set=name,v12=A(y,z),auc=res[name],d_vs_v12=res[name]-A(y,z),d_vs_base=res[name]-res.get('base',res[name]))
    print(json.dumps({k:(round(v,5) if isinstance(v,float) else v) for k,v in r.items()}),flush=True); open('exp_results.jsonl','a').write(json.dumps(r)+'\n')
print('t',time.time()-t0)
