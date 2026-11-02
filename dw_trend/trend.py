"""dw_trend: leak-safe 'trend family' features.  READ-ONLY on everything outside /workspace/huawei/dw_trend.
trend_feats(D, X, rows) -> {name: float32 array, one per row}.  Same contract as dw_v12sig/dw_p33 builders.
  D    : dw_signals.common.Data (for X=Jd(27) extended with eval events via dw_v12sig.sigs.extend_eval ONLY so own-history lookups work)
  X    : fit cutoff (unix s).  Item stats use train events ts < X only (rows+labels of those events) and the dedup click log D.snap(X).
  rows : dict(e=event idx per row, item=item idx per row, ts=event ts per row)
Per-row inputs: its own user's pre-ts history (D.OFF/HI/HT), candidate, ts. NO eval-row aggregates; item trends are measured AT THE CUTOFF X, not at row time.
Features (13):
 item  (at cutoff): it_tr_ratio   log(smoothed CTR last 1d / smoothed CTR prior 2d)
                    it_tr_slope   OLS slope over 3 daily smoothed CTRs / global CTR
                    it_imp_gr     log((imp last 1d+5)/(imp prior 2d/2+5))   impression growth
                    it_clk_gr     log((clicks last 1d+1)/(clicks prior 2d/2+1)) from the dedup click log (history clicks included)
                    it_hrs_clk    hours between item's last click and X (clip 240)
 user  (own hist) : u_rate_tr     log((clicks in last 24h+.5)/((clicks 24-96h before ts)/3+.5))
                    u_gap_tr      log(mean click gap of newest half / mean gap of oldest half of history)  (>=6 clicks)
                    u_overdue     log((ts-last click)/ mean gap)
 taste (p33 12-d PPMI-SVD space, own hist split newest-5 vs older, >=8 clicks):
                    tx_cos_rec, tx_cos_old, tx_dcos (=rec-old), tx_drift (1-cos(rec,old); user-level), tx_proj (cand . unit(rec-old))"""
import sys; sys.dont_write_bytecode=True
sys.path.insert(0,'/workspace/huawei/dw_p33'); sys.path.insert(0,'/workspace/huawei/dw_v12sig'); sys.path.insert(0,'/workspace/huawei/dw_signals')
import numpy as np, polars as pl
from scipy import sparse
from common import Data, Jd, DAY, J20
from sigs import extend_eval
import p33feat
CACHE='/workspace/huawei/dw_trend/cache'
def _unit(a): return a/(np.linalg.norm(a,axis=-1,keepdims=True)+1e-9)

def item_feats(D,X,S,item):
    nX=int(np.searchsorted(D.ETS,X)); NI=D.NI
    ts=D.ETS[:nX]; b=((X-ts)//DAY).astype(np.int64); ok=b<3
    RI=D.RI[:nX][ok]; Y=D.Y[:nX][ok]; bb=np.repeat(b[ok],10)
    imp=np.bincount(RI.ravel()*3+bb,minlength=NI*3).reshape(NI,3).astype(np.float64)
    clk=np.bincount(RI.ravel()[Y.ravel()==1]*3+bb[Y.ravel()==1],minlength=NI*3).reshape(NI,3).astype(np.float64)
    p0=clk.sum()/imp.sum(); m=20.0
    ctr=(clk+m*p0)/(imp+m)                         # (NI,3): bins 0=last day,1,2
    cP=(clk[:,1:].sum(1)+m*p0)/(imp[:,1:].sum(1)+m); cR=ctr[:,0]
    tr_ratio=np.log(cR/cP)
    t=np.array([1.,0.,-1.]); t=t-t.mean()      # day index increasing toward the cutoff: bin2 oldest -> bin0 newest
    slope=(ctr@t)/(t@t)/p0
    imp_gr=np.log((imp[:,0]+5)/(imp[:,1:].sum(1)/2+5))
    ct=S['ct']; ci=S['ci']; bl=((X-ct)//DAY)
    k0=bl==0; k1=(bl==1)|(bl==2)
    c0=np.bincount(ci[k0],minlength=NI); c1=np.bincount(ci[k1],minlength=NI)
    clk_gr=np.log((c0+1)/(c1/2+1))
    last=np.full(NI,-np.inf); np.maximum.at(last,ci,ct); hrs=np.where(np.isfinite(last),np.minimum((X-last)/3600,240),np.nan)
    tab=dict(it_tr_ratio=tr_ratio,it_tr_slope=slope,it_imp_gr=imp_gr,it_clk_gr=clk_gr,it_hrs_clk=hrs)
    return {k:v[item].astype(np.float32) for k,v in tab.items()}

def _hist_idx(D,ev):
    st=D.OFF[ev]; en=D.OFF[ev+1]; ln=en-st
    evx=np.repeat(np.arange(len(ev)),ln); idx=np.repeat(st-np.concatenate([[0],np.cumsum(ln)[:-1]]),ln)+np.arange(ln.sum())
    return st,en,ln,evx,idx

def user_feats(D,ev,tev):
    n=len(ev); st,en,ln,evx,idx=_hist_idx(D,ev); age=(tev[evx]-D.HT[idx]).astype(np.float64)
    assert (age>0).all()
    n24=np.bincount(evx,weights=(age<=DAY),minlength=n); n96=np.bincount(evx,weights=(age<=4*DAY),minlength=n)-n24
    rate=np.log((n24+.5)/(n96/3+.5))
    h=ln//2; ok=ln>=6; hs=np.maximum(h-1,1)
    last=np.where(ln>0,D.HT[np.maximum(en-1,0)],0).astype(np.float64)
    gr=(D.HT[np.maximum(en-1,0)]-D.HT[np.maximum(en-h,0)])/hs; go=(D.HT[np.maximum(st+h-1,0)]-D.HT[np.maximum(st,0)])/hs
    gap_tr=np.where(ok,np.log((gr+600)/(go+600)),np.nan)
    gm=(D.HT[np.maximum(en-1,0)]-D.HT[np.maximum(st,0)])/np.maximum(ln-1,1)
    overdue=np.where(ln>=3,np.log((tev-last+600)/(gm+600)),np.nan)
    return dict(u_rate_tr=rate,u_gap_tr=gap_tr,u_overdue=overdue)

def taste_feats(D,X,S,ev,item,E12,k=5,min_len=8):
    n=len(ev); st,en,ln,evx,idx=_hist_idx(D,ev); pos=np.repeat(en,ln)-1-idx; hi=D.HI[idx]
    En=_unit(E12); ev_rec=pos<k
    def H(m): 
        M=sparse.csr_matrix((np.ones(m.sum(),np.float32),(evx[m],hi[m])),shape=(n,D.NI)); M.sum_duplicates(); return M
    Rec=np.asarray(H(ev_rec)@En); Old=np.asarray(H(~ev_rec)@En)
    ok=ln>=min_len; rep=np.repeat(np.arange(n),10)
    Ru,Ou=_unit(Rec),_unit(Old); cand=En[item]; nz=np.linalg.norm(E12[item],axis=1)>1e-6
    cr=(Ru[rep]*cand).sum(1); co=(Ou[rep]*cand).sum(1); okr=ok[rep]&nz
    dirv=_unit(Ru-Ou)
    return dict(tx_cos_rec=np.where(okr,cr,np.nan),tx_cos_old=np.where(okr,co,np.nan),tx_dcos=np.where(okr,cr-co,np.nan),
                tx_drift=np.where(ok,1-(Ru*Ou).sum(1),np.nan)[rep],tx_proj=np.where(okr,(dirv[rep]*cand).sum(1),np.nan))

def trend_feats(D,X,rows,E12=None,groups=('item','user','taste')):
    S=D.snap(X); e=rows['e']; ev=e[::10]; item=rows['item']; rep=np.repeat(np.arange(len(ev)),10); out={}
    if 'item' in groups: out.update(item_feats(D,X,S,item))
    if 'user' in groups:
        u=user_feats(D,ev,D.ETS[ev]); out.update({k:v[rep] for k,v in u.items()})
    if 'taste' in groups:
        if E12 is None: E12=p33feat._space(D,S,20,'inv',1.0,16)[:,:12]
        out.update(taste_feats(D,X,S,ev,item,E12))
    return {k:np.asarray(v,np.float32) for k,v in out.items()}

if __name__=='__main__':
    import time,os; os.makedirs(CACHE,exist_ok=True)
    for c in [int(x) for x in sys.argv[1].split(',')]:
        t=time.time(); D=Data()
        if c==27: nt=D.NE; V=extend_eval(D); assert int(np.searchsorted(D.ETS,Jd(27)))==nt==686824
        else: V=D.val(c)
        F=trend_feats(D,Jd(c),V); df=pl.DataFrame(F).with_columns(pl.Series('rowid',V['rowid'])); df.write_parquet(f'{CACHE}/trend_c{c}.parquet')
        print(c,df.shape,round(time.time()-t),'s',flush=True)
        print(df.describe().to_pandas().T.to_string() if False else df.select(pl.all().null_count()).row(0),flush=True)
