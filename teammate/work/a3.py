import numpy as np, polars as pl, scipy.sparse as sp
from sklearn.utils.extmath import randomized_svd
D='/workspace/huawei/v3/data/'
x=pl.read_parquet('X.parquet')
nI=pl.read_parquet(D+'items.parquet')['ii'].max()+1
et=pl.read_parquet(D+'ev_train.parquet',columns=['u','ts','h'])
# user sets: latest train history per user (covers older clicks) -- train-only
uh=et.sort('ts').group_by('u').agg(pl.col('h').last()).explode('h').drop_nulls().unique()
B=sp.csr_matrix((np.ones(len(uh),np.float32),(uh['u'].to_numpy(),uh['h'].to_numpy())),shape=(uh['u'].max()+1,nI))
deg=np.asarray(B.sum(0)).ravel()
Bn=B.multiply(1/np.sqrt(np.maximum(deg,1))).tocsr()
C=(Bn.T@Bn).tocsr(); C.setdiag(0); C.eliminate_zeros(); print('C nnz',C.nnz)
ev=pl.read_parquet(D+'ev_eval.parquet',columns=['eid','h'])
# eval event x item history matrix (own history only)
eh=ev.with_row_index('k').explode('h').drop_nulls()
H=sp.csr_matrix((np.ones(len(eh),np.float32),(eh['k'].to_numpy(),eh['h'].to_numpy())),shape=(len(ev),nI))
kmap=dict(zip(ev['eid'].to_list(),range(len(ev))))
k=np.array([kmap[e] for e in x['eid'].to_list()]); ii=x["ii"].to_numpy().copy()
HC=(H@C).tocsr()
cc=np.asarray(HC[k,ii]).ravel(); hl=np.maximum(x['hlen'].to_numpy(),1)
feats={'cocl_sum':cc,'cocl_mean':cc/hl}
for d in [8,16,32,64]:
    U,S,Vt=randomized_svd(Bn,d,random_state=0); E=(Vt.T*np.sqrt(S)).astype(np.float32); En=E/np.maximum(np.linalg.norm(E,axis=1,keepdims=True),1e-6)
    hm=H@En; hm=hm/np.maximum(np.linalg.norm(hm,axis=1,keepdims=True),1e-6)
    feats[f'svd{d}_cos']=(hm[k]*En[ii]).sum(1)
x=x.with_columns(**{n:pl.Series(v) for n,v in feats.items()})
x.write_parquet('X2.parquet'); print('ok', list(feats))
