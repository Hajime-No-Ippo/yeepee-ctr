import sys, os, json, numpy as np, polars as pl, lightgbm as lgb, warnings; warnings.filterwarnings('ignore')
sys.dont_write_bytecode=True
sys.path.insert(0,'/workspace/huawei/dw_levers'); sys.path.insert(1,'/workspace/huawei/dw_signals')
from sklearn.metrics import roc_auc_score as A
from sklearn.model_selection import GroupKFold
H='/workspace/huawei'; DAY=86400; J20=1737331200
def Jd(d): return J20+(d-20)*DAY
def z12(c):
    s=lambda t,k:np.load(f'{H}/v12/ens/{t}_c{c}_s0.npy')[:,k]
    m=np.mean([np.load(f'{H}/v12/ens/dwmlp_c{c}_s{k}.npy')[:,3] for k in (range(3) if c!=26 else [0])],0)
    return ((s('dwbig',1)+s('dw_s',1)+2*m)/4).astype(np.float64)
def meta(c): return pl.read_parquet(f'{H}/v10/feat_v3/val{c}.parquet',columns=['rowid','eid','u','ts','clicked'])
def base_ctx(z):
    Z=z.reshape(-1,10); mu=Z.mean(1,keepdims=True); sd=Z.std(1,keepdims=True)+1e-6
    rk=Z.argsort(1).argsort(1).astype(np.float32)
    lse=np.log(np.exp(Z).sum(1,keepdims=True))
    cols=[z,((Z-mu)/sd).ravel(),(Z-mu).ravel(),rk.ravel(),np.repeat(lse,10,1).ravel(),np.repeat(Z.max(1,keepdims=True),10,1).ravel(),np.repeat(mu,10,1).ravel(),np.repeat(sd,10,1).ravel()]
    return np.column_stack(cols).astype(np.float32)
PI=dict(objective='binary',learning_rate=0.03,num_leaves=15,min_data_in_leaf=500,feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,lambda_l2=30,max_bin=127,verbose=-1,num_threads=4)
def oof(X,y,z,user,seed=0,rounds=200,k=5,params=None):
    p=np.zeros(len(y)); P={**PI,**(params or {})}
    for tr,te in GroupKFold(k).split(X,y,user):
        m=lgb.train({**P,'seed':seed},lgb.Dataset(X[tr],y[tr],init_score=z[tr]),rounds); p[te]=m.predict(X[te],raw_score=True)+z[te]
    return p
def boot_delta(y,p0,p1,ev,nb=200,seed=0):
    """bootstrap over events (10 rows each) of AUC(p1)-AUC(p0)"""
    rs=np.random.RandomState(seed); n=len(y)//10; Y=y.reshape(n,10); a=p0.reshape(n,10); b=p1.reshape(n,10); out=[]
    for _ in range(nb):
        i=rs.randint(0,n,n); yy=Y[i].ravel(); out.append(A(yy,b[i].ravel())-A(yy,a[i].ravel()))
    return np.percentile(out,[2.5,97.5]), np.std(out)
def load_cut(c):
    m=meta(c); z=z12(c); assert len(z)==m.height
    return dict(c=c,y=m['clicked'].to_numpy().astype(int),z=z,user=m['u'].to_numpy(),rowid=m['rowid'].to_numpy(),ts=m['ts'].to_numpy(),Xb=base_ctx(z))
def report(name,F,Xnew,seeds=(0,),log='results.jsonl',control_cols=None):
    """F: cut dict; Xnew: (n,k) new feats. control = Xb only; test = Xb+Xnew. returns delta vs z and vs control"""
    y,z,u=F['y'],F['z'],F['user']; res=dict(name=name,cut=F['c'],nfeat=Xnew.shape[1],base=A(y,z))
    pc=np.mean([oof(F['Xb'],y,z,u,s) for s in seeds],0); X=np.hstack([F['Xb'],Xnew]); pt=np.mean([oof(X,y,z,u,s) for s in seeds],0)
    res.update(control=A(y,pc),test=A(y,pt)); res['d_vs_z']=res['test']-res['base']; res['d_vs_ctrl']=res['test']-res['control']
    ci,sd=boot_delta(y,pc,pt,None); res['ci_vs_ctrl']=[float(ci[0]),float(ci[1])]; res['sd']=float(sd)
    ev=lambda p: A((y.reshape(-1,10).max(1)),np.log(np.exp(p.reshape(-1,10)).sum(1)))
    res['evAUC_base']=ev(z); res['evAUC_test']=ev(pt)
    print(json.dumps({k:(round(v,5) if isinstance(v,float) else v) for k,v in res.items()}),flush=True)
    open(f'/workspace/huawei/dw_lb/{log}','a').write(json.dumps(res)+'\n'); return res,pt
