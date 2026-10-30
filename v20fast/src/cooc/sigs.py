"""(v17run: trimmed copy of dw_v12sig/sigs.py, only extend_eval + _zwithin) reusable, leak-safe feature functions.  Signature: f(D, X, rows) -> {name: float32 array len(rows['item'])}
D: dw_signals/common.Data (train events; for eval cuts D is extended with eval events via extend_eval ONLY so own-history lookups work),
X: fit cutoff (unix s). Every aggregate uses train events with ts < X only (+ the click log rebuilt from them). Per-row inputs are the row's own
user, candidate, ts, own pre-ts history and own slate. No eval-row labels/impressions/histories are aggregated."""
import sys, os; sys.dont_write_bytecode=True
sys.path.insert(0,'@@WORK@@/dw')
import numpy as np, polars as pl
from common import *

def extend_eval(D):
    """copy of v12/mkdw.py:extend_eval (append eval events so own-history lookups work). Returns rows dict."""
    ee=pl.read_parquet('@@WORK@@/evaluation_events.parquet').sort('ts','eid'); er=pl.read_parquet('@@WORK@@/evaluation_rows.parquet')
    assert ee['ts'].min()>=D.ETS.max()
    n=ee.height; eid2e={x:i for i,x in enumerate(ee['eid'].to_list())}
    er=er.with_columns(pl.col('eid').replace_strict(eid2e).alias('_e')).sort('_e','rowid'); assert (np.bincount(er['_e'].to_numpy())==10).all()
    pos=np.minimum(np.searchsorted(D.uu,ee['user'].to_numpy()),D.NU-1); seen=D.uu[pos]==ee['user'].to_numpy()
    eu=np.where(seen,pos,D.NU).astype(np.int32)
    ri=np.searchsorted(D.items,er['item'].to_numpy()).astype(np.int32)
    hi=ee['h_items'].explode().drop_nulls().to_numpy(); ht=ee['h_ts'].explode().drop_nulls().to_numpy().astype(np.int64); hl=ee['h_items'].list.len().to_numpy()
    hp=np.minimum(np.searchsorted(D.items,hi),D.NI-1); ok=D.items[hp]==hi
    hl=hl.astype(np.int64); cs=np.concatenate([[0],np.cumsum(ok.astype(np.int64))]); ce=np.concatenate([[0],np.cumsum(hl)]); hl2=cs[ce[1:]]-cs[ce[:-1]]
    D0=D.NE; D.ETS=np.concatenate([D.ETS,ee['ts'].to_numpy()]); D.EU=np.concatenate([D.EU,eu]); D.Hlen=np.concatenate([D.Hlen,hl2.astype(np.int64)])
    base=D.OFF[-1]; D.OFF=np.concatenate([D.OFF[:-1],base+np.concatenate([[0],np.cumsum(hl2)])]).astype(np.int64)
    D.HI=np.concatenate([D.HI,hp[ok].astype(np.int32)]); D.HT=np.concatenate([D.HT,ht[ok]])
    D.RI=np.concatenate([D.RI,ri.reshape(n,10)]); D.Y=np.concatenate([D.Y,np.zeros((n,10),np.int8)]); D.NE=D0+n; D.NU=D.NU+1
    e=np.repeat(np.arange(D0,D0+n),10)
    return dict(rowid=er['rowid'].to_numpy(),e=e,item=ri,user=D.EU[e],ts=D.ETS[e])

def _zwithin(x):
    m=x.reshape(-1,10); mu=np.nanmean(m,1,keepdims=True); sd=np.nanstd(m,1,keepdims=True)+1e-6
    return ((m-mu)/sd).ravel().astype(np.float32),(m-mu).ravel().astype(np.float32)

