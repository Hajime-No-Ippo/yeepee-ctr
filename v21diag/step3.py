"""Step 3: slate-level P(click) model, trained out-of-time (same folds as v19), combined with v19 scores.
usage: step3.py build|fit CUT VARIANT(B|A) [threads]"""
import sys, os, json, time, gc
thr=int(sys.argv[4]) if len(sys.argv)>4 else 2
os.environ.update(OMP_NUM_THREADS=str(thr), POLARS_MAX_THREADS=str(thr))
sys.path.insert(0,'/workspace/huawei/v19'); os.chdir('/workspace/huawei/v21diag')
import numpy as np, lightgbm as lgb
from cl import matrix
mode,cut,v=sys.argv[1],int(sys.argv[2]),sys.argv[3]
BASE=('feat_lr2','feat_tt','feat_pp','feat_sqC','feat_dw'); extra=BASE+(f'feat_{v}p',f'feat_N{v}s')
trn=[f'f{k}' for k in range(max(1,cut-24),cut-20)]; vn='eval' if cut==27 else f'val{cut}'
def slate(name):
    X,m,f=matrix(name,1.0,variant='notime',extra=extra)
    y=m['clicked'].to_numpy() if 'clicked' in m.columns else np.zeros(len(m),np.int64); eid=m['eid'].to_numpy(); n=len(y); assert n%10==0
    E=n//10; assert (eid.reshape(E,10)==eid.reshape(E,10)[:,:1]).all(), 'slate blocks not contiguous'
    Xr=X.reshape(E,10,-1); del X
    mu=np.nanmean(Xr,1); mx=np.nanmax(Xr,1); sd=np.nanstd(Xr,1)
    F=np.concatenate([mu,mx,sd],1).astype(np.float32)
    names=[f'mu_{c}' for c in f]+[f'mx_{c}' for c in f]+[f'sd_{c}' for c in f]
    return F,y.reshape(E,10).max(1),names
if mode=='build':
    t=time.time(); Fs=[];ys=[]
    for n in trn+[vn]:
        F,h,names=slate(n); Fs.append(F); ys.append(h); print(n,F.shape,h.mean(),round(time.time()-t),flush=True); gc.collect()
    np.savez(f'ens/slate_c{cut}_{v}.npz',*Fs,*ys,names=np.array(names),k=len(trn))
else:
    d=np.load(f'ens/slate_c{cut}_{v}.npz',allow_pickle=True); k=int(d['k']); names=list(d['names'])
    Fs=[d[f'arr_{i}'] for i in range(k+1)]; ys=[d[f'arr_{i}'] for i in range(k+1,2*k+2)]
    X=np.concatenate(Fs[:k]); y=np.concatenate(ys[:k]); Xv,yv=Fs[k],ys[k]
    from sklearn.metrics import roc_auc_score as AUC
    keep=[i for i in range(X.shape[1]) if np.nanstd(X[:,i])>0]; X=X[:,keep]; Xv=Xv[:,keep]
    P=dict(objective='binary',learning_rate=0.03,num_leaves=31,min_data_in_leaf=100,feature_fraction=0.3,bagging_fraction=0.8,bagging_freq=1,lambda_l2=30,max_bin=63,verbose=-1,num_threads=thr)
    ds=lgb.Dataset(X,y); dv=lgb.Dataset(Xv,yv,reference=ds); ev={}
    m=lgb.train(P,ds,600,valid_sets=[dv],callbacks=[lgb.record_evaluation(ev)])
    out={r:(round(float(AUC(yv,m.predict(Xv,num_iteration=r,raw_score=True))),5) if cut!=27 else None) for r in (100,200,300,400,600)}
    print(json.dumps(dict(cut=cut,v=v,n=len(y),nf=X.shape[1],slateAUC_by_round=out)),flush=True)
    np.save(f'ens/pslate_c{cut}_{v}.npy',np.stack([m.predict(Xv,num_iteration=r,raw_score=True) for r in (200,400,600)],1).astype(np.float32))
    imp=sorted(zip(m.feature_importance('gain'),[names[i] for i in keep]))[::-1][:25]
    print([ (n,int(g)) for g,n in imp])
