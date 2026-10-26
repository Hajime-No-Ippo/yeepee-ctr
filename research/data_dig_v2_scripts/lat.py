# shared loader for dig v2 (pre-C=J25 train data, all users; eval on sampled val25/26 rows in D.parquet)
import polars as pl, numpy as np, scipy.sparse as sp, sys, lightgbm as lgb
from scipy.sparse.linalg import svds
from sklearn.metrics import roc_auc_score
sys.path.insert(0,'/tmp/dig'); from h import wauc
J20=1737331200; DAY=86400; C=J20+5*DAY
imap=pl.read_parquet('/tmp/dig/imap.parquet'); NI=imap.height
EV=pl.scan_parquet('/workspace/huawei/v2work/train_events.parquet').filter(pl.col('ts')<C)
LH=EV.sort('ts').group_by('user').agg(pl.col('h_items').last()).collect().with_row_index('u').select('user','u')
ALL=(EV.select('user','h_items','h_ts').explode('h_items','h_ts').unique(['user','h_items','h_ts']).collect()
     .rename({'h_items':'item','h_ts':'t'}).join(imap,on='item').join(LH,on='user').sort('u','t')
     .with_columns(pl.int_range(pl.len()).over('u').alias('p'),pl.len().over('u').alias('L')))
NU=LH.height
D=pl.read_parquet('/tmp/dig/D.parquet')
E=pl.read_parquet('/tmp/dig/events.parquet').select('eid','h_items','h_ts')
H=(D.select('eid').unique().join(E,on='eid').sort('eid').with_row_index('k').explode('h_items','h_ts')
   .rename({'h_items':'item','h_ts':'t'}).join(imap,on='item',how='left').drop_nulls('ix')
   .with_columns(pl.int_range(pl.len()).over('k').alias('pos'),pl.len().over('k').alias('Lh')))
kdf=H.select('eid','k').unique()
rk=D.select('eid').join(kdf,on='eid',how='left',maintain_order='left')['k'].fill_null(-1).to_numpy()
ix=D.join(imap,on='item',how='left',maintain_order='left')['ix'].to_numpy()
y=D['clicked'].to_numpy(); base=D['base'].to_numpy(); cut=D['cut'].to_numpy()
fold=(D['user'].hash(seed=3)%5).to_numpy()
def mat(df,alpha=0.5,beta=0.0):
    df=df.unique(['u','ix'])
    M=sp.csr_matrix((np.ones(df.height,np.float32),(df['u'].to_numpy(),df['ix'].to_numpy())),shape=(NU,NI))
    pop=np.asarray(M.sum(0)).ravel(); ln=np.asarray(M.sum(1)).ravel()
    return (sp.diags(1/np.maximum(ln,1)**beta)@M@sp.diags(1/(pop+5)**alpha)).tocsr()
def svdv(M,k):
    _,s,vt=svds(M,k=k,random_state=0); o=np.argsort(-s); return s[o],vt[o]
def nrm(V): return V/(np.linalg.norm(V,axis=1,keepdims=True)+1e-9)
def hcent(V,w=None):
    w=np.ones(H.height) if w is None else w
    hc=np.zeros((H['k'].max()+1,V.shape[1])); np.add.at(hc,H['k'].to_numpy(),V[H['ix'].to_numpy()]*w[:,None])
    return np.where(rk[:,None]>=0,hc[np.maximum(rk,0)],0)
def cosf(V,w=None):
    V=nrm(V); hcr=hcent(V,w); return (V[ix]*hcr).sum(1)/(np.linalg.norm(hcr,axis=1)+1e-9)
def ev(x,name='',res=True,seeds=2):
    dd=D.select('eid','clicked','cut').with_columns(pl.Series('x',x))
    g={c:round(roc_auc_score(y[cut==c],np.nan_to_num(x[cut==c],nan=-9)),4) for c in (25,26)}
    w={c:round(wauc(dd.filter(pl.col('cut')==c),'x'),4) for c in (25,26)}
    r=resid(x[:,None] if x.ndim==1 else x,seeds) if res else None
    print(f'{name:34s} glob {g} within {w} resid {r}',flush=True); return g,w,r
def resid(X,seeds=2):
    X=X if X.ndim==2 else X[:,None]; out={25:[],26:[]}
    for s in range(seeds):
        p=np.zeros(len(y))
        for f in range(5):
            tr=fold!=f; te=~tr
            m=lgb.train(dict(objective='binary',learning_rate=0.03,num_leaves=7,min_data_in_leaf=300,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10,verbose=-1,num_threads=2,seed=s),lgb.Dataset(X[tr],y[tr],init_score=base[tr]),150)
            p[te]=base[te]+m.predict(X[te],raw_score=True)
        for c in (25,26): out[c].append(roc_auc_score(y[cut==c],p[cut==c]))
    return {c:round(float(np.mean(v)-roc_auc_score(y[cut==c],base[cut==c])),4) for c,v in out.items()}
