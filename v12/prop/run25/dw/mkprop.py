"""mkdw.py NAME... : DW reusable feature families (impression_taste, slate_centrality, skip_contrast) for rows of fold/val/eval table NAME.
Fit cutoff X: f{k} -> J20+k*DAY; val23/25/26 -> J20+3/5/6 d; eval -> J27 (end of train). Only train events ts<X feed any aggregate; each row uses own history+own slate.
Writes feat_dw/NAME.parquet (rowid + 42 float32 cols). dw_signals is imported read-only (no bytecode writes)."""
import sys, os, time, json
os.environ['PYTHONDONTWRITEBYTECODE']='1'; sys.dont_write_bytecode=True
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import numpy as np, polars as pl
from common import *
W='/workspace/huawei/v12/sol_work/clean25/work'; OUT=f'{W}/feat_prop'; F3=f'{W}/feat_v3'; os.makedirs(OUT,exist_ok=True)
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

def build(d,C,v):
    e=v['e'][::10]; nE=len(e)
    ec=d.Y.max(1).astype(np.float32); ts=d.ETS; u=d.EU
    nX=int(np.searchsorted(ts,C))  # events before C
    # per-user aggregates over events with ts<C
    n_ev=np.bincount(u[:nX],minlength=d.NU).astype(np.float32); n_cl=np.bincount(u[:nX],weights=ec[:nX],minlength=d.NU)
    first=np.full(d.NU,np.inf); np.minimum.at(first,u[:nX],ts[:nX]); last=np.full(d.NU,-np.inf); np.maximum.at(last,u[:nX],ts[:nX])
    # last-3 events click rate before C, trailing no-click streak
    o=np.lexsort((ts[:nX],u[:nX])); us=u[:nX][o]; ecs=ec[:nX][o]; tss=ts[:nX][o]
    b=np.r_[0,np.flatnonzero(np.diff(us))+1,len(us)]
    l3=np.full(d.NU,np.nan,np.float32); streak=np.zeros(d.NU,np.float32); gapm=np.full(d.NU,np.nan,np.float32); gapsd=np.full(d.NU,np.nan,np.float32)
    for a,bb in zip(b[:-1],b[1:]):
        uu=us[a]; seg=ecs[a:bb]; l3[uu]=seg[-3:].mean(); r=0
        for x in seg[::-1]:
            if x==0: r+=1
            else: break
        streak[uu]=r
        if bb-a>=2: g=np.diff(tss[a:bb]); gapm[uu]=g.mean(); gapsd[uu]=g.std()
    ue=u[e]; tev=ts[e]
    shrink=(n_cl[ue]+0.63*3)/(n_ev[ue]+3)
    days=(C-J20)/DAY
    col={}
    col['p_nev']=n_ev[ue]; col['p_evrate']=n_ev[ue]/days; col['p_ctr']=np.where(n_ev[ue]>0,n_cl[ue]/np.maximum(n_ev[ue],1),np.nan); col['p_ctr_s']=shrink
    col['p_l3']=l3[ue]; col['p_streak']=streak[ue]; col['p_gapm']=gapm[ue]/3600; col['p_gapsd']=gapsd[ue]/3600
    col['p_since_ev']=np.where(np.isfinite(last[ue]),(tev-last[ue])/3600,np.nan); col['p_since_first']=np.where(np.isfinite(first[ue]),(tev-first[ue])/3600,np.nan)
    col['p_gap_ratio']=col['p_since_ev']/(col['p_gapm']+1)
    # own history
    hl=d.Hlen[e]; col['h_len']=hl.astype(np.float32)
    hmin=np.full(nE,np.nan); hmax=np.full(nE,np.nan); hgm=np.full(nE,np.nan); hgsd=np.full(nE,np.nan); hglast=np.full(nE,np.nan); h1=np.zeros(nE); h3=np.zeros(nE); h7=np.zeros(nE); hwarm=np.zeros(nE); hwin=np.zeros(nE); huniq=np.zeros(nE)
    for i,ee in enumerate(e):
        a,bb=d.OFF[ee],d.OFF[ee+1]
        if bb-a==0: continue
        t=d.HT[a:bb]; it=d.HI[a:bb]; tt=tev[i]
        hmin[i]=(tt-t[0])/3600; hmax[i]=(tt-t[-1])/3600; huniq[i]=len(set(it.tolist()))
        h1[i]=(t>tt-DAY).sum(); h3[i]=(t>tt-3*DAY).sum(); h7[i]=(t>tt-7*DAY).sum(); hwarm[i]=(t<J20).sum(); hwin[i]=(t>=J20).sum()
        if bb-a>=2: g=np.diff(t); hgm[i]=g.mean()/3600; hgsd[i]=g.std()/3600; hglast[i]=g[-1]/3600
    col.update(h_span=hmin,h_since_last=hmax,h_gap_mean=hgm,h_gap_sd=hgsd,h_gap_last=hglast,h_1d=h1,h_3d=h3,h_7d=h7,h_warm=hwarm,h_win=hwin,h_uniq=huniq,h_uniq_ratio=huniq/np.maximum(hl,1))
    col['h_rate_day']=np.where(hl>=2,(hl-1)/np.maximum(hmin-hmax,1e-3)*24,np.nan)
    # clicks per event: window clicks (in train, ts<C) / events
    col['p_clk_per_ev']=n_cl[ue]/np.maximum(n_ev[ue],1)
    # history density per unit time vs event rate: implied clicks per event = click_rate/event_rate
    erate=np.where(n_ev[ue]>0,n_ev[ue]/np.maximum((C-J20)/3600/24,1e-3),np.nan)  # events per day
    col['implied_cpe']=col['h_rate_day']/(erate+0.5)
    col['gap_over_hgap']=col['p_gapm']/(col['h_gap_mean']+1)
    X=np.column_stack([np.repeat(np.asarray(col[k],np.float32),10) for k in col]).astype(np.float32)
    return X,list(col)

if __name__=='__main__':
    D=Data(); names=sys.argv[1:]
    for nm in names:
        t=time.time(); X=CUT[nm]
        if nm=='eval':
            nt=D.NE; V=extend_eval(D); assert int(np.searchsorted(D.ETS,X))==nt
        else: V=rows_train(D,nm)
        A,names_=build(D,X,V); df=pl.DataFrame({k:A[:,i] for i,k in enumerate(names_)}).with_columns(pl.Series('rowid',V['rowid'])).select(['rowid']+names_)
        assert df.height==len(V['rowid']); df.write_parquet(f'{OUT}/{nm}.parquet'); print(nm,df.shape,f'{round(time.time()-t)}s',flush=True)
