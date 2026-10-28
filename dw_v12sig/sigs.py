"""dw_v12sig reusable, leak-safe feature functions.  Signature: f(D, X, rows) -> {name: float32 array len(rows['item'])}
D: dw_signals/common.Data (train events; for eval cuts D is extended with eval events via extend_eval ONLY so own-history lookups work),
X: fit cutoff (unix s). Every aggregate uses train events with ts < X only (+ the click log rebuilt from them). Per-row inputs are the row's own
user, candidate, ts, own pre-ts history and own slate. No eval-row labels/impressions/histories are aggregated."""
import sys, os; sys.dont_write_bytecode=True
sys.path.insert(0,'/workspace/huawei/dw_signals')
import numpy as np, polars as pl
from common import *

def extend_eval(D):
    """copy of v12/mkdw.py:extend_eval (append eval events so own-history lookups work). Returns rows dict."""
    ee=pl.read_parquet('/workspace/huawei/lite/evaluation_events.parquet').sort('ts','eid'); er=pl.read_parquet('/workspace/huawei/lite/evaluation_rows.parquet')
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

# ---------- 1. item choice-model strength (conditional logit with outside option, MM fit) ----------
def fit_item_strength(D,X,lo=0.0,hi_iter=40,a=3.0):
    """P(click i | slate)=g_i/(1+sum g_j); P(no click)=1/(1+sum g). MM updates g_i=(c_i+a*g0)/(sum_{e∋i}1/D_e + a). events with lo<=ts<X."""
    nX=int(np.searchsorted(D.ETS,X)); s=int(np.searchsorted(D.ETS,lo)) if lo>0 else 0
    S=D.RI[s:nX]; Y=D.Y[s:nX]; ci=S[Y.astype(bool)]
    c=np.bincount(ci,minlength=D.NI).astype(np.float64); g=np.full(D.NI,0.07); g0=0.07
    for _ in range(hi_iter):
        De=1+g[S].sum(1); w=np.bincount(S.ravel(),weights=np.repeat(1/De,10),minlength=D.NI)
        g=(c+a*g0)/(w+a); g0=g.mean() if False else float(np.median(g[c>0])) if (c>0).any() else g0
    ni=np.bincount(S.ravel(),minlength=D.NI)
    return g.astype(np.float32),ni

def item_strength(D,X,rows):
    out={}; S=D.RI[rows['e'][::10]]
    for tag,lo,a in (('all',0.0,3.0),('all_s20',0.0,20.0),('l48',X-2*DAY,5.0)):
        g,ni=fit_item_strength(D,X,lo,a=a); lg=np.log(g)
        r=lg[rows['item']]; z,dm=_zwithin(r); out[f'cl_{tag}_lg']=r.astype(np.float32); out[f'cl_{tag}_z']=z; out[f'cl_{tag}_dm']=dm
        De=np.repeat(np.log(1+g[S].sum(1)),10); out[f'cl_{tag}_logD']=De.astype(np.float32)   # model-implied event any-click level (item-strength only)
        out[f'cl_{tag}_mx']=np.repeat(lg[S].max(1),10).astype(np.float32)
    return out

# ---------- 2. item lifecycle ----------
def item_lifecycle(D,X,rows):
    nX=int(np.searchsorted(D.ETS,X)); S=D.RI[:nX]; ts=D.ETS[:nX]
    first=np.full(D.NI,np.inf); last=np.full(D.NI,-np.inf)
    tsr=np.repeat(ts,10); np.minimum.at(first,S.ravel(),tsr); np.maximum.at(last,S.ravel(),tsr)
    # also first click in history (warm-up) is NOT impression info; use only impressions
    it=rows['item']; t=rows['ts'].astype(np.float64); out={}
    age=(t-first[it])/3600; age=np.where(np.isfinite(age),age,-1.0); out['lc_age_h']=age.astype(np.float32); out['lc_logage']=np.log1p(np.maximum(age,0)).astype(np.float32)
    out['lc_new']=(~np.isfinite(first[it])).astype(np.float32)
    gap=(t-last[it])/3600; out['lc_gap_last_imp_h']=np.where(np.isfinite(gap),gap,np.nan).astype(np.float32)
    # CTR lift as a function of item age (global curve from train events <X, age at event time)
    Yc=D.Y[:nX]; ag=(tsr-first[S.ravel()])/3600; ab=np.minimum(np.floor(np.log2(1+np.maximum(ag,0))*2).astype(int),40)
    num=np.bincount(ab,weights=Yc.ravel(),minlength=41); den=np.bincount(ab,minlength=41)+1e-9; curve=(num+5*0.063)/(den+5)
    rb=np.minimum(np.floor(np.log2(1+np.maximum(age,0))*2).astype(int),40); out['lc_agecurve']=np.where(age>=0,curve[rb],np.nan).astype(np.float32)
    # item impression volume in the last 6/24h before X relative to its lifetime rate
    r24=np.bincount(S[ts>=X-DAY].ravel(),minlength=D.NI); r6=np.bincount(S[ts>=X-6*3600].ravel(),minlength=D.NI); tot=np.bincount(S.ravel(),minlength=D.NI)
    life=np.maximum((X-first)/3600,1.0); out['lc_rel24']=np.where(np.isfinite(first[it]),(r24[it]/24)/(tot[it]/life[it]+1e-3),np.nan).astype(np.float32)
    out['lc_rel6']=np.where(np.isfinite(first[it]),(r6[it]/6)/(tot[it]/life[it]+1e-3),np.nan).astype(np.float32)
    return out

# ---------- 3. sequence: next-click transitions + per-item repeat propensity ----------
def _pair_counts(D,snap,lag):
    u,i,t=snap['cu'],snap['ci'],snap['ct']; o=np.lexsort((t,u)); u=u[o]; i=i[o]
    m=u[lag:]==u[:-lag]; a=i[:-lag][m].astype(np.int64); b=i[lag:][m].astype(np.int64)
    keys,cnt=np.unique(a*D.NI+b,return_counts=True); return keys,cnt
def _lookup(keys,cnt,q):
    p=np.minimum(np.searchsorted(keys,q),len(keys)-1); return np.where(keys[p]==q,cnt[p],0).astype(np.float32)
def sequence(D,X,rows):
    snap=D.snap(X); out={}; e=rows['e']; it=rows['item'].astype(np.int64); n=len(it)
    nout=np.bincount(snap['ci'],minlength=D.NI).astype(np.float32); tot=float(len(snap['ci']))
    hl=D.Hlen[e]; end=D.OFF[e+1]
    for lag in (1,2,3):
        keys,cnt=_pair_counts(D,snap,lag)
        src_n=np.bincount((keys//D.NI).astype(np.int64),weights=cnt,minlength=D.NI).astype(np.float32)   # transitions out of item
        for k in range(1,lag+1) if lag==1 else (lag,):
            # k-th last history item -> candidate  (lag-step transition)
            h=np.where(hl>=k,D.HI[np.maximum(end-k,0)],-1).astype(np.int64); ok=h>=0
            c=_lookup(keys,cnt,np.where(ok,h*D.NI+it,0)); nh=np.where(ok,src_n[np.maximum(h,0)],0)
            pmi=np.log((c+0.5)/(nh+5)*tot/(nout[it]+5))
            out[f'sq_l{lag}_k{k}_cnt']=np.where(ok,c,np.nan).astype(np.float32); out[f'sq_l{lag}_k{k}_pmi']=np.where(ok,pmi,np.nan).astype(np.float32)
            out[f'sq_l{lag}_k{k}_pmi_z']=_zwithin(out[f'sq_l{lag}_k{k}_pmi'])[0]
            # reverse direction: candidate -> h  (item preceded by candidate => candidate is an 'antecedent' of last click)
            cr=_lookup(keys,cnt,np.where(ok,it*D.NI+h,0)); out[f'sq_l{lag}_k{k}_rev']=np.where(ok,cr,np.nan).astype(np.float32)
    # max transition pmi from any of last 5 history items (lag-1 sense: unordered co-click within 5 steps of the user's own log)
    # per-item repeat propensity
    u,i,t=snap['cu'],snap['ci'],snap['ct']; o=np.lexsort((t,i,u)); u=u[o]; i=i[o]; same=np.r_[False,(u[1:]==u[:-1])&(i[1:]==i[:-1])]
    rep=np.bincount(i[same],minlength=D.NI).astype(np.float32); allc=np.bincount(i,minlength=D.NI).astype(np.float32)
    rr=(rep+0.5)/(allc+8); out['rp_item_rate']=rr[it].astype(np.float32)
    # own-history repeat context
    A=D.HI; cnt_own=np.zeros(n,np.float32); lastgap=np.full(n,np.nan,np.float32)
    ev=e[::10]; 
    for k,ee in enumerate(ev):
        a,b=D.OFF[ee],D.OFF[ee+1]
        if b>a:
            hi=A[a:b]; slate=rows['item'][k*10:(k+1)*10]
            for j,c in enumerate(slate):
                m=hi==c
                if m.any(): cnt_own[k*10+j]=m.sum(); lastgap[k*10+j]=(rows['ts'][k*10]-D.HT[a:b][m][-1])/3600
    out['rp_own_cnt']=cnt_own; out['rp_own_gap_h']=lastgap; out['rp_x_rate']=(cnt_own>0)*out['rp_item_rate']
    return out

def all_new(D,X,rows):
    F={}; F.update(item_strength(D,X,rows)); F.update(item_lifecycle(D,X,rows)); F.update(sequence(D,X,rows))
    return pl.DataFrame({k:np.asarray(v,np.float32) for k,v in F.items()})

if __name__=='__main__':
    import time; c=int(sys.argv[1]); D=Data(); t=time.time()
    if c==27:
        nt=D.NE; V=extend_eval(D); assert int(np.searchsorted(D.ETS,Jd(27)))==nt
    else: V=D.val(c)
    df=all_new(D,Jd(c),V); df=df.with_columns(pl.Series('rowid',V['rowid'])); df.write_parquet(f'/workspace/huawei/dw_v12sig/cache/sig_c{c}.parquet'); print(c,df.shape,round(time.time()-t),'s',flush=True)
