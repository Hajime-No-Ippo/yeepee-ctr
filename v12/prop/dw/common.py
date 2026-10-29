"""DW signals shared loader (vendored from dw_signals/common.py; paths -> work dir).
Snapshot(C): everything known from train events with ts < C (rows+labels of those events, plus click log reconstructed from their histories).
"""
import numpy as np, polars as pl, time, gc
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import os
L=os.environ.get('DW_WORK','@@WORK@@'); DAY=86400; J20=1737331200   # L = work dir holding train_events/train_rows/evaluation_* parquet (prep.py output)
def Jd(d): return J20+(d-20)*DAY
T0=time.time()
def log(*a): print(f'[{time.time()-T0:6.0f}s]',*a,flush=True)

class Data:
    def __init__(s):
        ev=pl.read_parquet(f'{L}/train_events.parquet').sort('ts').with_row_index('e')
        rows=pl.read_parquet(f'{L}/train_rows.parquet').join(ev.select('eid','e'),on='eid').sort(['e','item'])
        s.NE=ev.height; assert rows.height==s.NE*10
        s.ETS=ev['ts'].to_numpy(); s.EID=ev['eid'].to_numpy()
        uu,s.EU=np.unique(ev['user'].to_numpy(),return_inverse=True); s.EU=s.EU.astype(np.int32); s.NU=len(uu); s.uu=uu
        s.Hlen=ev['h_items'].list.len().to_numpy().astype(np.int64)
        s.OFF=np.zeros(s.NE+1,np.int64); s.OFF[1:]=np.cumsum(s.Hlen)
        hi=ev['h_items'].explode().drop_nulls().to_numpy(); s.HT=ev['h_ts'].explode().drop_nulls().to_numpy().astype(np.int64)
        rr=rows['item'].to_numpy()
        s.items=np.unique(np.concatenate([np.unique(rr),np.unique(hi)])); s.NI=len(s.items)
        s.HI=np.searchsorted(s.items,hi).astype(np.int32); s.RI=np.searchsorted(s.items,rr).astype(np.int32).reshape(s.NE,10)
        s.Y=rows['clicked'].to_numpy().astype(np.int8).reshape(s.NE,10)
        s.rowid=rows['rowid'].to_numpy().reshape(s.NE,10)
        s.eid2e={}  # not needed
        s.ev=ev.select('e','eid','user','ts'); s.rows_rowid=rows.select('rowid','e','item')
        del ev,rows; gc.collect()
    def snap(s,X):
        """click log (user,item,ts) from events with ts<X (dedup) + counts. returns dict"""
        nX=int(np.searchsorted(s.ETS,X))
        hu=np.repeat(s.EU[:nX],s.Hlen[:nX]); hi=s.HI[:s.OFF[nX]]; ht=s.HT[:s.OFF[nX]]
        ce,cj=np.nonzero(s.Y[:nX]); 
        cu=np.concatenate([hu,s.EU[ce]]); ci=np.concatenate([hi,s.RI[ce,cj]]); ct=np.concatenate([ht,s.ETS[ce]])
        tm=(ct-Jd(1))//60
        key=np.unique((cu.astype(np.int64)*s.NI+ci)*(1<<20)+tm)
        ui=key>>20; tm=key&((1<<20)-1)
        return dict(nX=nX,cu=(ui//s.NI).astype(np.int32),ci=(ui%s.NI).astype(np.int32),ct=tm*60+Jd(1))

def click_svd(S,D,k=32):
    """item/user taste vectors from dedup click log (binary, 1/sqrt((nu+1)(ni+1)) scaling). returns item vec (V*s) L2-normed, user vec"""
    u,i=S['cu'],S['ci']; key=np.unique(u.astype(np.int64)*D.NI+i); uu=(key//D.NI).astype(np.int32); ii=(key%D.NI).astype(np.int32)
    nu=np.bincount(uu,minlength=D.NU).astype(np.float32); ni=np.bincount(ii,minlength=D.NI).astype(np.float32)
    M=csr_matrix((1/np.sqrt((nu[uu]+1)*(ni[ii]+1)),(uu,ii)),shape=(D.NU,D.NI),dtype=np.float32)
    U,sv,Vt=svds(M,k=k,random_state=0)
    V=(Vt.T*sv); V/= np.linalg.norm(V,axis=1,keepdims=True)+1e-9
    return V.astype(np.float32),(U*sv).astype(np.float32)
def norm(x): return x/(np.linalg.norm(x,axis=-1,keepdims=True)+1e-9)
def event_view(a): return a.reshape(-1,10)
def within_event(x):
    """z-score, demeaned, rank (0..9 ascending) within the 10-row events; x aligned with val rows"""
    m=x.reshape(-1,10); mu=np.nanmean(m,1,keepdims=True); sd=np.nanstd(m,1,keepdims=True)+1e-6
    rk=m.argsort(1).argsort(1).astype(np.float32)
    return ((m-mu)/sd).ravel(),(m-mu).ravel(),rk.ravel()
