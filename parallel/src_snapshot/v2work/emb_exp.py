import polars as pl, numpy as np, sys
sys.path.insert(0,'/workspace/huawei/v2work')
from feats import load, first_events
from scipy import sparse
from sklearn.utils.extmath import randomized_svd
from sklearn.metrics import roc_auc_score
TE,TR,EE,ER,NI=load('/workspace/huawei/v2work')
DAY=86400; C=1737331200+4*DAY
T_ev=first_events(TE,C,DAY); T_rows=TR.join(T_ev.select('eid'),on='eid')
P_ev=TE.filter(pl.col('ts')<C); P=TR.join(P_ev.select('eid','user','ts'),on='eid')
hc=lambda E: E.select('user','h','h_ts').explode(['h','h_ts']).drop_nulls().rename({'h':'ii','h_ts':'ts'})
pool=pl.concat([hc(P_ev),hc(T_ev),P.filter(pl.col('clicked')==1).select('user','ii','ts')]).unique()
us=pool.group_by('user','ii').agg(pl.len().alias('n'))
uidx=us.select('user').unique().with_row_index('ur'); m=us.join(uidx,on='user')
R=T_rows.join(T_ev.select('eid','user','h'),on='eid').sort('eid')
y=R['clicked'].to_numpy(); eid=R['eid'].to_numpy()
ex=R.with_row_index('r').select('r','h').with_columns(pl.int_ranges(pl.col('h').list.len(),0,-1).alias('pos')).explode(['h','pos']).drop_nulls()
rr=ex['r'].to_numpy(); hh=ex['h'].to_numpy(); pos=ex['pos'].to_numpy()
um=R.select('user').join(uidx,on='user',how='left')['ur'].fill_null(0).to_numpy()
def dm(x):
    s=pl.DataFrame({'e':eid,'x':x}).with_columns((pl.col('x')-pl.col('x').mean().over('e')).alias('d'))['d'].to_numpy(); return s
def ev(name,V,U=None,decay=0.85):
    hv=np.zeros((len(R),V.shape[1]),np.float32); np.add.at(hv,rr,V[hh]*(decay**(pos-1))[:,None])
    hvn=hv/(np.linalg.norm(hv,axis=1,keepdims=True)+1e-6); c=V[R['ii'].to_numpy()]
    s=(hvn*c).sum(1); out=f'{name}: hist cos {roc_auc_score(y,s):.4f} dm {roc_auc_score(y,dm(s)):.4f}'
    if U is not None:
        s2=(U[um]*c).sum(1); out+=f' | ucos {roc_auc_score(y,s2):.4f} dm {roc_auc_score(y,dm(s2)):.4f}'
    print(out,flush=True)
ii=m['ii'].to_numpy(); ur=m['ur'].to_numpy()
df_=np.bincount(ii,minlength=NI); idf=np.log((len(uidx)+1)/(df_+1)).astype(np.float32)
for wname,w in [('bin',np.ones(len(ii),np.float32)),('idf',idf[ii])]:
    X=sparse.csr_matrix((w,(ur,ii)),shape=(len(uidx),NI)); rn=np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel())+1e-6; X=sparse.diags(1/rn)@X
    for k in (128,256):
        U,S,Vt=randomized_svd(X,k,random_state=0,n_iter=4)
        for a in (0.0,):
            V=(Vt.T*S**a).astype(np.float32); Vn=V/(np.linalg.norm(V,axis=1,keepdims=True)+1e-6)
            Uf=(U*S**a).astype(np.float32); Uf/=np.linalg.norm(Uf,axis=1,keepdims=True)+1e-6
            ev(f'{wname} k={k} a={a}',Vn,Uf)
