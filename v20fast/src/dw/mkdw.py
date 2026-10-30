"""mkdw.py NAME... : DW reusable feature families (impression_taste, slate_centrality, skip_contrast) for rows of fold/val/eval table NAME.
Fit cutoff X: f{k} -> J20+k*DAY; val23/25/26 -> J20+3/5/6 d; eval -> J27 (end of train). Only train events ts<X feed any aggregate; each row uses own history+own slate.
Writes feat_dw/NAME.parquet (rowid + 42 float32 cols). dw_signals is imported read-only (no bytecode writes)."""
import sys, os, time, json
os.environ['PYTHONDONTWRITEBYTECODE']='1'; sys.dont_write_bytecode=True
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import numpy as np, polars as pl
from common import *
from reusable import all_recommended
W='@@WORK@@'; OUT=f'{W}/feat_dw'; F3=f'{W}/feat_v3'; os.makedirs(OUT,exist_ok=True)
CUT={**{f'f{k}':J20+k*DAY for k in range(1,7)},'val23':J20+3*DAY,'val25':J20+5*DAY,'val26':J20+6*DAY,'eval':J20+7*DAY}
def rows_train(D,name):
    v=pl.read_parquet(f'{F3}/{name}.parquet',columns=['rowid'])
    m=v.join(D.rows_rowid,on='rowid',how='left',maintain_order='left'); assert m['e'].null_count()==0
    e=m['e'].to_numpy().astype(np.int64); it=np.searchsorted(D.items,m['item'].to_numpy()).astype(np.int32)
    assert (e.reshape(-1,10)==e.reshape(-1,10)[:,:1]).all()
    return dict(rowid=v['rowid'].to_numpy(),e=e,item=it,user=D.EU[e],ts=D.ETS[e])
def extend_eval(D):
    """append eval events (after all train events, ts sorted) to D's arrays so history lookups work; unseen users -> one empty extra user row."""
    ee=pl.read_parquet(f'{W}/evaluation_events.parquet').sort('ts','eid'); er=pl.read_parquet(f'{W}/evaluation_rows.parquet')
    assert ee['ts'].min()>=D.ETS.max()
    n=ee.height; eid2e={x:i for i,x in enumerate(ee['eid'].to_list())}
    er=er.with_columns(pl.col('eid').replace_strict(eid2e).alias('_e')).sort('_e','rowid'); assert (np.bincount(er['_e'].to_numpy())==10).all()
    pos=np.searchsorted(D.uu,ee['user'].to_numpy()); pos=np.minimum(pos,D.NU-1); seen=D.uu[pos]==ee['user'].to_numpy()
    new_u=D.NU; eu=np.where(seen,pos,new_u).astype(np.int32); print('eval unseen users',int((~seen).sum()),flush=True)
    ri=np.searchsorted(D.items,er['item'].to_numpy()); assert (D.items[ri]==er['item'].to_numpy()).all(); ri=ri.astype(np.int32)
    hi=ee['h_items'].explode().drop_nulls().to_numpy(); ht=ee['h_ts'].explode().drop_nulls().to_numpy().astype(np.int64); hl=ee['h_items'].list.len().to_numpy()
    hp=np.minimum(np.searchsorted(D.items,hi),D.NI-1); ok=D.items[hp]==hi; print('eval hist items unseen (dropped)',int((~ok).sum()),'of',len(hi),flush=True)
    hl=hl.astype(np.int64); cs=np.concatenate([[0],np.cumsum(ok.astype(np.int64))]); ce=np.concatenate([[0],np.cumsum(hl)]); hl2=cs[ce[1:]]-cs[ce[:-1]]
    D0=D.NE; D.ETS=np.concatenate([D.ETS,ee['ts'].to_numpy()]); D.EU=np.concatenate([D.EU,eu]); D.Hlen=np.concatenate([D.Hlen,hl2.astype(np.int64)])
    base=D.OFF[-1]; D.OFF=np.concatenate([D.OFF[:-1],base+np.concatenate([[0],np.cumsum(hl2)])]).astype(np.int64)
    D.HI=np.concatenate([D.HI,hp[ok].astype(np.int32)]); D.HT=np.concatenate([D.HT,ht[ok]])
    D.RI=np.concatenate([D.RI,ri.reshape(n,10)]); D.Y=np.concatenate([D.Y,np.zeros((n,10),np.int8)]); D.NE=D0+n; D.NU=D.NU+1
    assert len(D.OFF)==D.NE+1
    e=np.repeat(np.arange(D0,D0+n),10)
    return dict(rowid=er['rowid'].to_numpy(),e=e,item=ri,user=D.EU[e],ts=D.ETS[e])
if __name__=='__main__':
    D=Data(); names=sys.argv[1:]
    for nm in names:
        t=time.time(); X=CUT[nm]
        if nm=='eval':
            nt=D.NE; V=extend_eval(D); assert int(np.searchsorted(D.ETS,X))==nt   # snapshot = exactly the train events
        else: V=rows_train(D,nm)
        df=all_recommended(D,X,V); df=df.with_columns(pl.Series('rowid',V['rowid'])).select(['rowid']+[c for c in df.columns])
        df.write_parquet(f'{OUT}/{nm}.parquet'); s=round(time.time()-t); print(nm,df.shape,f'{s}s',flush=True)
