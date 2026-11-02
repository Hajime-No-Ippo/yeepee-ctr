"""TRANSDUCTIVE trend features (clearly labelled TT_*; Eric lifted the eval-statistics ban for performance, 2026-11-02).
Uses UNLABELLED structure of the evaluation window's own rows: the slates (items) and timestamps of the cut's val rows (= each user's first event >= X),
for validation cuts exactly the same construction on the fold's own val rows (never their labels, never their histories beyond own-history lookups).
tt_feats(D,X,rows) -> {name: float32}.   Features:
 TT_imp_ratio   log((n_val_imp(item)+3)/(expected from train last-1d impression rate scaled to #val events + 3))   eval-window popularity vs recent train
 TT_imp_loc     log((val impressions of item within +-6h of row ts +1)/(n_val_imp * frac_of_val_events_within_6h +1))   local-time popularity burst
 TT_slate_pop   mean over the event's 10 candidates of log(n_val_imp+1)  (slate composition, event level)
 TT_coimp_mean/TT_coimp_max   PMI of candidate with the user's last-5 history items in the val co-impression matrix (items impressed together in val slates)
TT_*_dm = within-event demeaned versions are built by the harness."""
import sys; sys.dont_write_bytecode=True
sys.path.insert(0,'/workspace/huawei/dw_trend')
from trend import *

def tt_feats(D,X,rows):
    e=rows['e']; ev=e[::10]; Nv=len(ev); item=rows['item']; tsr=rows['ts']; NI=D.NI; rep=np.repeat(np.arange(Nv),10)
    vi=np.bincount(item,minlength=NI).astype(np.float64)
    nX=int(np.searchsorted(D.ETS,X)); ts=D.ETS[:nX]; m=(X-ts)<DAY; NR=m.sum()
    imp0=np.bincount(D.RI[:nX][m].ravel(),minlength=NI).astype(np.float64)
    out={}
    out['TT_imp_ratio']=np.log((vi+3)/(imp0*Nv/NR+3))[item]
    # local-time burst
    BIG=1<<40; key=np.sort(item.astype(np.int64)*BIG+tsr); k=item.astype(np.int64)*BIG+tsr; w=6*3600
    nloc=np.searchsorted(key,k+w,side='right')-np.searchsorted(key,k-w,side='left')
    evt=np.sort(D.ETS[ev]); fr=(np.searchsorted(evt,D.ETS[ev]+w,side='right')-np.searchsorted(evt,D.ETS[ev]-w,side='left'))/Nv
    out['TT_imp_loc']=np.log((nloc+1)/(vi[item]*fr[rep]+1))
    out['TT_slate_pop']=np.repeat(np.log(vi[item]+1).reshape(-1,10).mean(1),10)
    # val co-impression PMI
    it=item.reshape(Nv,10); a=np.repeat(it,10,1).ravel(); b=np.tile(it,(1,10)).ravel(); k_=a!=b
    M=sparse.coo_matrix((np.ones(k_.sum(),np.float32),(a[k_],b[k_])),shape=(NI,NI)).tocsr(); M.sum_duplicates()
    r=np.asarray(M.sum(1)).ravel(); T=M.sum()
    P=M.tocoo(); pm=np.log(P.data*T/(r[P.row]*r[P.col]+1e-9)); P=sparse.csr_matrix((np.maximum(pm,0).astype(np.float32),(P.row,P.col)),shape=(NI,NI))
    st,en,ln,evx,idx=p33feat_hist(D,ev)
    pos=np.repeat(en,ln)-1-idx; sel=pos<5; evs=evx[sel]; hs=D.HI[idx][sel]
    # per row x history item
    rr=(evs[:,None]*10+np.arange(10)[None,:]).ravel(); cand=item[rr]; hh=np.repeat(hs,10)
    vals=np.asarray(P[cand,hh]).ravel()
    s=np.bincount(rr,weights=vals,minlength=Nv*10); mx=np.zeros(Nv*10); np.maximum.at(mx,rr,vals); cnt=np.bincount(rr,minlength=Nv*10)
    has=np.repeat(ln>0,10)
    out['TT_coimp_mean']=np.where(has,s/np.maximum(cnt,1),np.nan); out['TT_coimp_max']=np.where(has,mx,np.nan)
    return {k:np.asarray(v,np.float32) for k,v in out.items()}
def p33feat_hist(D,ev):
    from trend import _hist_idx; return _hist_idx(D,ev)
if __name__=='__main__':
    import time
    for c in [int(x) for x in sys.argv[1].split(',')]:
        t=time.time(); D=Data()
        if c==27: nt=D.NE; V=extend_eval(D); assert int(np.searchsorted(D.ETS,Jd(27)))==nt==686824
        else: V=D.val(c)
        F=tt_feats(D,Jd(c),V); df=pl.DataFrame(F).with_columns(pl.Series('rowid',V['rowid'])); df.write_parquet(f'{CACHE}/tt_c{c}.parquet')
        print(c,df.shape,round(time.time()-t),'s',df.select(pl.all().null_count()).row(0),flush=True)
