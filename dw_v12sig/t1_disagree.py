import sys; sys.dont_write_bytecode=True
import numpy as np, polars as pl, lightgbm as lgb
from scipy.stats import norm, rankdata, spearmanr
from sklearn.model_selection import GroupKFold
H='/workspace/huawei'
ev=pl.read_parquet(f'{H}/v10/feat_v3/eval.parquet')
v12=pl.read_csv(f'{H}/v12/out/submission_v12.csv').rename({'predicted_probability':'v12'})
p33=pl.read_parquet(f'{H}/teammate/work33/p33_recovered_ranks.parquet')
d=ev.join(v12,on='rowid').join(p33,on='rowid'); assert d.height==ev.height
n=d.height
nz=lambda r: norm.ppf((rankdata(r)-0.5)/len(r))
a=nz(d['v12'].to_numpy()); b=nz(d['p33_rank'].to_numpy())
print('spearman',spearmanr(a,b)[0],'pearson normal scores',np.corrcoef(a,b)[0,1])
# residual of b on a (linear)
beta=np.cov(a,b)[0,1]/a.var(); res=b-beta*a; print('beta',beta,'resid sd',res.std())
R=res.reshape(-1,10); assert (d['eid'].to_numpy().reshape(-1,10)==d['eid'].to_numpy().reshape(-1,10)[:,:1]).all()
evm=R.mean(1,keepdims=True); within=(R-evm).ravel(); evl=np.repeat(evm,10,1).ravel()
print('var share event-level',evl.var()/res.var(),'within',within.var()/res.var())
d=d.with_columns(pl.Series('res',res),pl.Series('res_ev',evl),pl.Series('res_w',within),pl.Series('za',a),pl.Series('zb',b))
d.select(['rowid','res','res_ev','res_w','za','zb']).write_parquet('t1_resid.parquet')
# segments: abs residual and spearman by segment
def seg(col,bins,name):
    x=d[col].to_numpy(); q=np.digitize(x,bins); out=[]
    for k in np.unique(q):
        m=q==k; out.append((k,m.sum(),np.corrcoef(a[m],b[m])[0,1],np.abs(res[m]).mean(),res[m].mean(),np.abs(within[m]).mean(),np.abs(evl[m]).mean()))
    print('##',name); print('bin n corr mean|res| mean_res mean|within| mean|event|')
    for o in out: print('  %d %7d %.3f %.3f %+.3f %.3f %.3f'%o)
d=d.with_columns(pl.col('hist_n_all').fill_null(-1))
seg('us_hlen',[1,3,6,11,16,21,30],'hist len'); seg('tm_stale',[1,6,12,24,48,72],'stale h from first-cutoff?')
seg('hist_since_clk',[1,6,24,72,168],'hours since last hist click'); seg('it_imp',[20,50,100,200,400,1000],'item impressions')
seg('it_ctr',[0.03,0.05,0.07,0.1,0.15],'item ctr'); seg('ord',[1,3,5,7,9],'ord (slate pos?)'); seg('us_nev',[0,1,3,6,10],'user n events'); seg('hist_in',[0.5],'cand in own history')
seg('it_nclk_warm',[0,1,5,20,100],'item warm clicks'); seg('it_trend24',[-0.5,-0.1,0.1,0.5],'item trend')
# modelling: what features explain the residual?
feats=[c for c in ev.columns if c not in ('rowid','eid','ii','u')]
X=d.select(feats).to_numpy().astype(np.float32); user=d['u'].to_numpy()
for tgt,nm in ((res,'res'),(evl,'res_event'),(within,'res_within')):
    p=np.zeros(n)
    for tr,te in GroupKFold(5).split(X,tgt,user):
        m=lgb.train(dict(objective='regression',learning_rate=0.05,num_leaves=31,min_data_in_leaf=200,verbose=-1,num_threads=3,feature_fraction=0.7),lgb.Dataset(X[tr],tgt[tr]),250); p[te]=m.predict(X[te])
    print(nm,'R2 oof from %d feats'%len(feats),1-((tgt-p)**2).mean()/tgt.var())
    if nm=='res':
        imp=sorted(zip(m.feature_importance('gain'),feats),reverse=True)[:15]; print([ (f,int(g)) for g,f in imp])
# item-level random effect: split events by parity, correlate item mean residual (within-event)
item=d['ii'].to_numpy(); evi=np.repeat(np.arange(n//10),10); par=(evi%2)
df=pl.DataFrame(dict(item=item,par=par,w=within,r=res))
g=df.group_by(['item','par']).agg(pl.col('w').mean().alias('w'),pl.col('w').len().alias('n')).pivot(index='item',on='par',values=['w','n']).drop_nulls()
print(g.columns); c=g.columns
m=(g[c[3]]>=5)&(g[c[4]]>=5) if len(c)>4 else None
gg=g.filter(m); print('items with >=5 rows both halves',gg.height,'split-half corr of item mean within-residual',np.corrcoef(gg[c[1]],gg[c[2]])[0,1])
