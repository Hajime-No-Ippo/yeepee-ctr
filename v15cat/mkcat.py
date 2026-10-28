"""mkcat.py CUT : categorical feature tables for CatBoost, CUT in {25,23}. Cluster geometry (impression-SVD k=24, label-free) is fit ONCE per cut at
X = J20+5d (cut25) / J20+3d (cut23) = the val cutoff, from train events with ts<X (impressions only: no click labels, no eval rows), so cluster ids are
consistent across train folds and val. Writes /workspace/huawei/v15cat/cat{CUT}/NAME.parquet (rowid + int32 cols)."""
import sys, os, time
os.environ.setdefault('DW_WORK','/workspace/huawei/v12x/dwdata'); sys.path.insert(0,'/workspace/huawei/v12x/mk'); sys.dont_write_bytecode=True
import numpy as np, polars as pl
from sklearn.cluster import MiniBatchKMeans
from mkfeat import *
CUTN=int(sys.argv[1]); TABS=['f1','f2','f3','f4','val25'] if CUTN==25 else ['f1','f2','val23']
X=J20+(5 if CUTN==25 else 3)*DAY
D=Data(); log('data')
U,sv,Vv,nu=imp_svd(D,X); Un=norm(U*sv); Vn=norm(Vv*sv)
ni=np.bincount(D.RI[:int(np.searchsorted(D.ETS,X))].ravel(),minlength=D.NI)
def km(Z,seen,k):
    lab=np.full(len(Z),-1,np.int32); idx=np.where(seen)[0]
    lab[idx]=MiniBatchKMeans(k,random_state=0,n_init=3,batch_size=8192,max_iter=100).fit_predict(Z[idx]).astype(np.int32); return lab
IC={k:km(Vn,ni>0,k) for k in (64,256,1024)}; UC={k:km(Un,nu>0,k) for k in (64,256,1024)}; log('kmeans done')
def lg(x): return np.floor(np.log2(1+np.maximum(x,0))).astype(np.int32)
os.makedirs(f'/workspace/huawei/v15cat/cat{CUTN}',exist_ok=True)
for nm in TABS:
    V=rows_train(D,nm); e=V['e']; it=V['item']; us=np.minimum(V['user'],len(nu)-1); useen=V['user']<len(nu)
    tm=pl.read_parquet(f'{F3}/{nm}.parquet',columns=['rowid','tm_tb','tm_dt','tm_stale','us_hlen']); assert (tm['rowid'].to_numpy()==V['rowid']).all()
    c={'item_id':it.astype(np.int32),'user_id':V['user'].astype(np.int32)}
    for k in IC: c[f'ic{k}']=IC[k][it]
    for k in UC: c[f'uc{k}']=np.where(useen,UC[k][us],-1).astype(np.int32)
    ue=e[::10]; last=np.full(len(ue),-1,np.int64); rec=np.full(len(ue),-1,np.int64); inh=np.zeros(len(e),np.int32)
    for j,ee in enumerate(ue):
        a,b=D.OFF[ee],D.OFF[ee+1]
        if b>a: p=a+int(np.argmax(D.HT[a:b])); last[j]=D.HI[p]; rec[j]=D.ETS[ee]-D.HT[p]
        if b>a: h=set(D.HI[a:b].tolist()); inh[j*10:(j+1)*10]=[int(x in h) for x in it[j*10:(j+1)*10]]
    last=np.repeat(last,10); rec=np.repeat(rec,10)
    for k in (64,256): c[f'lastc{k}']=np.where(last>=0,IC[k][np.maximum(last,0)],-2).astype(np.int32)
    tb=tm['tm_tb'].fill_null(-1).to_numpy().astype(np.int32); dt=tm['tm_dt'].fill_null(-1).to_numpy().astype(np.int32)
    c['rec_bin']=np.where(rec>=0,lg(rec/60.),-1).astype(np.int32); c['hlen_bin']=lg(tm['us_hlen'].fill_null(0).to_numpy()); c['stale_bin']=lg(tm['tm_stale'].fill_null(0).to_numpy()/60.)
    c['time_bucket']=tb; c['day_type']=dt; c['in_hist']=inh
    c['x_ic256_uc256']=(c['ic256']+1)*1100+(c['uc256']+1); c['x_ic256_tb']=(c['ic256']+1)*100+tb; c['x_inh_ic256']=(c['ic256']+1)*2+inh
    c['x_ic64_uc64']=(c['ic64']+1)*100+(c['uc64']+1); c['x_lastc64_ic64']=(c['lastc64']+3)*100+(c['ic64']+1)
    df=pl.DataFrame({k:v.astype(np.int64 if k.startswith('x_') else np.int32) for k,v in c.items()}).with_columns(pl.Series('rowid',V['rowid'])).select(['rowid']+list(c))
    df.write_parquet(f'/workspace/huawei/v15cat/cat{CUTN}/{nm}.parquet'); log(nm,df.shape)
