"""exp1: headroom of event-level 'any click' model on top of v12 logits (OOF 5-fold by user on val rows)."""
import sys, json, time, numpy as np, polars as pl, lightgbm as lgb, warnings; warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score as A
from sklearn.model_selection import KFold
from cl import matrix, BASE
def z12(c):
    s=lambda t,k:np.load(f'/workspace/huawei/v12/ens/{t}_c{c}_s0.npy')[:,k]
    m=np.mean([np.load(f'/workspace/huawei/v12/ens/dwmlp_c{c}_s{k}.npy')[:,3] for k in (range(3) if c!=26 else [0])],0)
    return ((s('dwbig',1)+s('dw_s',1)+2*m)/4).astype(np.float64)
cut=int(sys.argv[1]); t0=time.time()
X,m,fl=matrix(f'val{cut}',1.0,extra=BASE+('feat_sqC','feat_dw')); y=m['clicked'].to_numpy(); z=z12(cut); print('v12',A(y,z),flush=True)
n=len(y)//10; X3=X.reshape(n,10,-1); yev=y.reshape(n,10).max(1); Z=z.reshape(n,10)
const=(np.nanmax(X3,1)-np.nanmin(X3,1)==0).all(0)|np.isnan(X3[:,0,:]).all(0)  # event-constant cols (approx)
print('event-const cols',const.sum(),'of',len(fl),flush=True)
C=X3[:,0,const]; V=X3[:,:,~const]
mean=np.nanmean(V,1); mx=np.nanmax(V,1); mn=np.nanmin(V,1); sd=np.nanstd(V,1)
p=1/(1+np.exp(-Z)); pany=1-np.prod(1-p,1); e0=np.log(pany/(1-pany))
zs=np.sort(Z,1)[:,::-1]; zst=np.column_stack([np.log(np.exp(Z).sum(1)),Z.max(1),Z.mean(1),Z.std(1),zs[:,0]-zs[:,1],zs[:,1]-zs[:,2],e0])
print('event AUC of e0',A(yev,e0),'lse',A(yev,zst[:,0]),flush=True)
sets={'zstats+const+agg':np.hstack([zst,C,mean,mx,mn,sd]),'zstats+const':np.hstack([zst,C])}
P=dict(objective='binary',learning_rate=0.03,num_leaves=7,min_data_in_leaf=300,feature_fraction=0.5,bagging_fraction=0.8,bagging_freq=1,lambda_l2=30,max_bin=63,verbose=-1,num_threads=8,seed=0)
for name,F in sets.items():
    for init in (True,):
        oof=np.zeros(n)
        for tr,te in KFold(5,shuffle=True,random_state=0).split(F):
            ds=lgb.Dataset(F[tr],yev[tr],init_score=e0[tr]); mdl=lgb.train(P,ds,120); oof[te]=mdl.predict(F[te],raw_score=True)+e0[te]
        d=oof-e0; r=dict(cut=cut,set=name,ev_auc_e0=A(yev,e0),ev_auc_new=A(yev,oof))
        for a in (0.1,0.25,0.5):
            zn=(Z+a*d[:,None]).ravel(); r[f'glob_a{a}']=A(y,zn)-A(y,z)
        print(json.dumps({k:(round(v,5) if isinstance(v,float) else v) for k,v in r.items()}),flush=True)
print('t',time.time()-t0)
