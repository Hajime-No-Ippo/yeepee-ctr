import polars as pl, numpy as np, scipy.sparse as sp, sys, lightgbm as lgb
from scipy.sparse.linalg import svds
from sklearn.metrics import roc_auc_score
sys.path.insert(0,'/tmp/dig'); from h import wauc
J20=1737331200;D_=86400;C=J20+5*D_
imap=pl.read_parquet('imap.parquet'); NI=imap.height
EV=pl.scan_parquet('/workspace/huawei/v2work/train_events.parquet').filter(pl.col('ts')<C)
lh=EV.sort('ts').group_by('user').agg(pl.col('h_items').last()).collect().with_row_index('u').explode('h_items').rename({'h_items':'item'}).join(imap,on='item')
cnt=lh.group_by('u','ix').len()
D=pl.read_parquet('D.parquet'); E=pl.read_parquet('events.parquet').select('eid','h_items')
H=D.select('eid').unique().join(E,on='eid').with_row_index('k').explode('h_items').rename({'h_items':'item'}).join(imap,on='item',how='left').drop_nulls('ix')
H=H.with_columns(pl.int_range(pl.len()).over('k').alias('pos'),pl.len().over('k').alias('L'))
kmap=dict(zip(*[H.unique('k').sort('k')[c].to_list() for c in ('eid','k')]))
rk=np.array([kmap.get(e,-1) for e in D['eid'].to_list()]); ix=D.join(imap,on='item',how='left')['ix'].to_numpy()
y=D['clicked'].to_numpy(); base=D['base'].to_numpy(); cut=D['cut'].to_numpy()
def feat(k,alpha,binar,recent):
    v=np.ones(cnt.height,np.float32) if binar else cnt['len'].to_numpy().astype(np.float32)
    M=sp.csr_matrix((v,(cnt['u'].to_numpy(),cnt['ix'].to_numpy())),shape=(cnt['u'].max()+1,NI))
    pop=np.asarray((M>0).sum(0)).ravel()
    M=M.multiply(1/(pop+5)**alpha).tocsr()
    _,s,vt=svds(M,k=k,random_state=0); V=vt.T*s; V/=np.linalg.norm(V,axis=1,keepdims=True)+1e-9
    w=np.exp(-(H['L']-1-H['pos']).to_numpy()/recent) if recent else np.ones(H.height)
    hc=np.zeros((H['k'].max()+1,k)); np.add.at(hc,H['k'].to_numpy(),V[H['ix'].to_numpy()]*w[:,None])
    hcr=np.where(rk[:,None]>=0,hc[np.maximum(rk,0)],0)
    return (V[ix]*hcr).sum(1)/(np.linalg.norm(hcr,axis=1)+1e-9)
fold=(D['user'].hash(seed=3)%5).to_numpy()
def resid(x,seeds=2):
    out={25:[],26:[]}
    X=x[:,None]
    for s in range(seeds):
        p=np.zeros(len(y))
        for f in range(5):
            tr=fold!=f; te=~tr
            m=lgb.train(dict(objective='binary',learning_rate=0.03,num_leaves=7,min_data_in_leaf=300,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10,verbose=-1,num_threads=2,seed=s),lgb.Dataset(X[tr],y[tr],init_score=base[tr]),150)
            p[te]=base[te]+m.predict(X[te],raw_score=True)
        for c in (25,26): out[c].append(roc_auc_score(y[cut==c],p[cut==c]))
    return {c:round(np.mean(v)-roc_auc_score(y[cut==c],base[cut==c]),5) for c,v in out.items()}
res={}
for cfg in [(32,0.5,1,0),(32,0,1,0),(32,1.0,1,0),(16,0.5,1,0),(64,0.5,1,0),(128,0.5,1,0),(32,0.5,0,0),(32,0.5,1,5),(64,0.5,1,5)]:
    x=feat(*cfg); dd=D.with_columns(pl.Series('x',x))
    sa={c:round(roc_auc_score(y[cut==c],x[cut==c]),4) for c in (25,26)}
    wa={c:round(wauc(dd.filter(pl.col('cut')==c),'x'),4) for c in (25,26)}
    r=resid(x); print('k,alpha,binary,recent',cfg,'glob',sa,'within',wa,'resid delta',r,flush=True)
    np.save(f'x_{"_".join(map(str,cfg))}.npy',x)
