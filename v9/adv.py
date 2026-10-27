"""Adversarial validation (train-fold rows vs eval rows) on features w/o time-since / cumulative counts.
Outputs ens/w_adv_{table}.npy (eval-likeness odds, per matrix row) for f1..f6 and val23/25/26."""
import numpy as np, polars as pl, lightgbm as lgb, gc, json, time, re
from cl import matrix
from sklearn.metrics import roc_auc_score
TAGS='ni' if __import__('os').environ.get('NOITEM') else ''; t0=time.time(); EX=('feat_lr2','feat_tt','feat_pp')
BAD=['since','nev','nclk','it_imp','it_clk','_imp','_clk','imp_rate','nusers','nuniq','uniq','hlen','h_len','_ev','up_n','ip_n','ip_users','ip_1d','ip_3d',
     'ip_dec','ip_warm','up_1d','ui_','uip_','hist_n','hist_noclk','hist_pos','hist_in','co_cnt','co_sum','inh_','climp','gap','1d_before','repeat','cold','nohist',
     'trend','co_nz','clk_dec','ip_per','tm_','_tb','_dt','hour','stale']+(['it_','v2_i_','v2_ip','v2_ev_i','clctr','us_ctr','u_ctr','mf_b'] if __import__('os').environ.get('NOITEM') else [])
TR=['f1','f2','f3','f4','f5','f6']; rng=np.random.default_rng(0)
def sub(n,frac):
    X,m,f=matrix(n,1.0,extra=EX); e=m['eid'].to_numpy(); ue=np.unique(e); keep=np.isin(e,rng.choice(ue,int(len(ue)*frac),replace=False))
    return X,m,f,keep
X,m,f,k=sub('eval',1.0); cols=[c for c in f if not any(b in c for b in BAD)]; idx=[f.index(c) for c in cols]
print('n feats',len(cols),'excluded',len(f)-len(cols),flush=True)
Xe=X[:, idx]; ge=m['eid'].to_numpy(); del X,m; gc.collect()
Xs=[];gs=[]
for n in TR:
    X,m,_,k=sub(n,0.16); Xs.append(X[k][:,idx]); gs.append(m['eid'].to_numpy()[k]); del X,m; gc.collect()
Xt=np.concatenate(Xs); gt=np.concatenate(gs); del Xs
XA=np.concatenate([Xt,Xe]); y=np.r_[np.zeros(len(Xt)),np.ones(len(Xe))]; g=np.r_[gt,ge+10**9]; del Xt,Xe; gc.collect()
print('rows train/eval',int((y==0).sum()),int(y.sum()),flush=True)
ue=np.unique(g); fm=dict(zip(ue,rng.integers(0,4,len(ue)))); fo=np.array([fm[e] for e in g])
P=dict(objective='binary',learning_rate=0.1,num_leaves=31,min_data_in_leaf=500,feature_fraction=0.8,bagging_fraction=0.7,bagging_freq=1,lambda_l2=10,max_bin=63,verbose=-1,num_threads=4)
oof=np.zeros(len(y)); imp=np.zeros(len(cols)); models=[]
for k in range(4):
    tr=fo!=k; mdl=lgb.train(P,lgb.Dataset(XA[tr],y[tr]),150); oof[~tr]=mdl.predict(XA[~tr]); imp+=mdl.feature_importance('gain'); models.append(mdl)
auc=roc_auc_score(y,oof); o=np.argsort(-imp)[:15]; top=[(cols[i],round(imp[i]/imp.sum(),3)) for i in o]
print('adv AUC',round(auc,4),'top',top,flush=True)
del XA; gc.collect()
prior=(y==0).sum()/y.sum(); res=dict(auc=auc,top=top,nfeat=len(cols))
for n in TR+['val23','val25','val26']:
    X,m,_=matrix(n,1.0,extra=EX); X=X[:,idx]; p=np.mean([md.predict(X) for md in models],0); p=np.clip(p,1e-4,1-1e-4)
    w=p/(1-p)*prior; np.save(f'ens/w_adv{TAGS}_{n}.npy',w.astype(np.float32))
    res[n]=dict(mean_p=float(p.mean()),w_p50=float(np.median(w)),w_p99=float(np.percentile(w,99)),ess=float(w.sum()**2/(w**2).sum()/len(w)))
    print(n,res[n],flush=True); del X,m; gc.collect()
json.dump(res,open(f'ens/adv{TAGS}_res.json','w'),indent=1); print('total s',round(time.time()-t0))
