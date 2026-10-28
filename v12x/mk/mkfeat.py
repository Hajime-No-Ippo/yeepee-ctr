"""mkfeat.py NAME... : builds, for fold/val tables NAME (rows of v12/feat_v3/NAME.parquet), at the fold's fit cutoff X (f{k}->J20+k*DAY, val23/25 -> J20+3/5 d),
 feat_prop (DW click-propensity, 28 cols), feat_tm (tm_prod0-11 + tm_prod_tail: per-dim impression-SVD taste products, k=24),
 feat_tmx (interaction: row's candidate already in the event's own history x taste: tm_inh, tm_cos, tm_cos_in, tm_cos_out, tm_cos_dm, tm_inh_n).
Only events ts<X feed any aggregate (as DW sigs2.imp_svd). Writes to /workspace/huawei/v12x/{feat_prop,feat_tm,feat_tmx}/NAME.parquet"""
import sys, os, time
os.environ.setdefault('DW_WORK','/workspace/huawei/lite'); sys.dont_write_bytecode=True; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import numpy as np, polars as pl
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from common import *
from propbuild import build
O='/workspace/huawei/v12x'; F3='/workspace/huawei/v12/feat_v3'
CUT={**{f'f{k}':J20+k*DAY for k in range(1,7)},'val23':J20+3*DAY,'val25':J20+5*DAY,'val26':J20+6*DAY}
def rows_train(D,name):
    v=pl.read_parquet(f'{F3}/{name}.parquet',columns=['rowid'])
    m=v.join(D.rows_rowid,on='rowid',how='left',maintain_order='left'); assert m['e'].null_count()==0
    e=m['e'].to_numpy().astype(np.int64); it=np.searchsorted(D.items,m['item'].to_numpy()).astype(np.int32)
    assert (e.reshape(-1,10)==e.reshape(-1,10)[:,:1]).all()
    return dict(rowid=v['rowid'].to_numpy(),e=e,item=it,user=D.EU[e],ts=D.ETS[e])
def imp_svd(D,X,k=24):   # identical to dw_v12sig/sigs2.py
    nX=int(np.searchsorted(D.ETS,X)); u=np.repeat(D.EU[:nX],10); it=D.RI[:nX].ravel()
    M=csr_matrix((np.ones(len(u),np.float32),(u,it)),shape=(D.NU,D.NI)); M.sum_duplicates()
    nu=np.asarray(M.sum(1)).ravel(); ni=np.asarray(M.sum(0)).ravel(); Mc=M.tocoo()
    Ms=csr_matrix((Mc.data/np.sqrt((nu[Mc.row]+1)*(ni[Mc.col]+1)),(Mc.row,Mc.col)),shape=M.shape,dtype=np.float32)
    U,sv,Vt=svds(Ms,k=k,random_state=0); o=np.argsort(-sv); U,sv,Vt=U[:,o],sv[o],Vt[o]
    return U,sv,Vt.T,nu
def tm(D,X,V):
    U,sv,Vv,nu=imp_svd(D,X); Un=norm(U*sv); Vn=norm(Vv*sv); u=V['user']; it=V['item']
    has=(nu[np.minimum(u,len(nu)-1)]>0)&(u<len(nu)); uu=np.where(has[:,None],Un[np.minimum(u,len(nu)-1)],0); P=uu*Vn[it]; out={}
    for d in range(12): out[f'tm_prod{d}']=np.where(has,P[:,d],np.nan).astype(np.float32)
    out['tm_prod_tail']=np.where(has,P[:,12:].sum(1),np.nan).astype(np.float32)
    # interaction with 'candidate in the event's own history'
    e=V['e']; ue,inv=np.unique(e[::10],return_inverse=True)
    keys=[]
    for ee in ue: keys.append(ee*D.NI+D.HI[D.OFF[ee]:D.OFF[ee+1]].astype(np.int64))
    keys=np.unique(np.concatenate(keys)); q=e*D.NI+it.astype(np.int64); pos=np.minimum(np.searchsorted(keys,q),len(keys)-1); inh=(keys[pos]==q).astype(np.float32)
    cos=np.where(has,P.sum(1),np.nan).astype(np.float32); g=cos.reshape(-1,10)
    x={'tm_inh':inh,'tm_cos':cos,'tm_cos_in':np.where(inh>0,cos,np.nan),'tm_cos_out':np.where(inh>0,np.nan,cos),
       'tm_cos_dm':(g-np.nanmean(g,1,keepdims=True)).ravel().astype(np.float32),'tm_inh_n':np.repeat(inh.reshape(-1,10).sum(1),10)}
    return out,x
if __name__=='__main__':
    D=Data(); log('data loaded')
    for nm in sys.argv[1:]:
        t=time.time(); X=CUT[nm]; V=rows_train(D,nm)
        A,pn=build(D,X,V); d1=pl.DataFrame({k:A[:,i] for i,k in enumerate(pn)}).with_columns(pl.Series('rowid',V['rowid'])).select(['rowid']+pn)
        o,x=tm(D,X,V); d2=pl.DataFrame(o).with_columns(pl.Series('rowid',V['rowid'])).select(['rowid']+list(o)); d3=pl.DataFrame(x).with_columns(pl.Series('rowid',V['rowid'])).select(['rowid']+list(x))
        for sub,df in (('feat_prop',d1),('feat_tm',d2),('feat_tmx',d3)):
            os.makedirs(f'{O}/{sub}',exist_ok=True); df.write_parquet(f'{O}/{sub}/{nm}.parquet')
        print(nm,d1.shape,d2.shape,d3.shape,round(time.time()-t),'s',flush=True)
