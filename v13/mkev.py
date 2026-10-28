"""feat_ev: event-slate aggregates (mean/max/min/std over the 10 own-slate rows) + row deviation for top-K features.
K selected by within-event label corr on f1 (train fold, earlier than every cut's eval). Uses only each table's own slates."""
import sys, json, os, time, numpy as np, polars as pl, gc
from cl import matrix, BASE
K=int(os.environ.get('K',80)); os.makedirs('feat_ev',exist_ok=True)
EX=BASE+('feat_sqC','feat_dw')
def sel():
    if os.path.exists('feat_ev/top.json'): return json.load(open('feat_ev/top.json'))
    X,m,fl=matrix('f1',1.0,extra=EX); y=m['clicked'].to_numpy().astype(np.float32); n=len(y)//10
    X3=X.reshape(n,10,-1); const=(np.nanmax(X3,1)-np.nanmin(X3,1)==0).all(0)|np.isnan(X3[:,0,:]).all(0)
    yc=(y.reshape(n,10)-y.reshape(n,10).mean(1,keepdims=True)).ravel(); sc=np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        if const[j]: continue
        v=X3[:,:,j]; d=np.nan_to_num(v-np.nanmean(v,1,keepdims=True)).ravel(); sc[j]=abs((d*yc).sum())/(np.sqrt((d**2).sum())+1e-9)
    top=[fl[i] for i in np.argsort(-sc)[:K]]; json.dump(top,open('feat_ev/top.json','w')); return top
top=sel(); print(top[:20],flush=True)
for name in sys.argv[1:]:
    t0=time.time(); X,m,fl=matrix(name,1.0,extra=EX); idx=[fl.index(c) for c in top]; V=X[:,idx].reshape(-1,10,len(idx)); del X; gc.collect()
    mu=np.nanmean(V,1,keepdims=True); mx=np.nanmax(V,1,keepdims=True); mn=np.nanmin(V,1,keepdims=True); sd=np.nanstd(V,1,keepdims=True)
    rep=lambda a:np.repeat(a[:,0,:],10,0)
    mu,mx,mn,sd=rep(mu),rep(mx),rep(mn),rep(sd); mu3=mu
    cols={'rowid':m['rowid'].to_numpy()}
    for k,c in enumerate(top):
        cols[f'ev_mean_{c}']=mu[:,k].copy(); cols[f'ev_max_{c}']=mx[:,k].copy(); cols[f'ev_min_{c}']=mn[:,k].copy(); cols[f'ev_std_{c}']=sd[:,k].copy()
        cols[f'ev_dm_{c}']=(V[:,:,k].ravel()-mu[:,k])
    pl.DataFrame({k:(v.astype(np.float32) if k!='rowid' else v) for k,v in cols.items()}).write_parquet(f'feat_ev/{name}.parquet'); print(name,len(m),time.time()-t0,flush=True)
