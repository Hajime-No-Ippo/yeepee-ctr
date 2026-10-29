"""dw_p33 reusable, leak-safe feature builder (teammate-p33-style): history-co-occurrence PPMI -> SVD(12/16) item embeddings,
history-centroid user vectors (uniform / recency-weighted), cosine, per-dim coordinate products, item/user coordinates, multi-window fits.
Signature: pp_feats(D, X, rows) -> {name: float32 array len(rows['item'])}; same contract as dw_v12sig/sigs2.taste_metric
(D = dw_signals.common.Data (extend_eval'ed for X=Jd(27)); X = fit cutoff; stats from train events ts<X only; per-row inputs: own user history/candidate/slate)."""
import sys; sys.dont_write_bytecode=True
sys.path.insert(0,'/workspace/huawei/dw_p33')
from p33lib import *
from sigs import _zwithin

def _unit(a): return a/(np.linalg.norm(a,axis=-1,keepdims=True)+1e-9)
def _space(D,S,W=20,wm='inv',alpha=1.0,k=16,seed=0):
    P=ppmi(click_pairs_matrix(D,S,W,wm),alpha,0.0); U,s=svd_embed(P,k+8,n_iter=7,seed=seed); return U[:,:k]   # whitened (p=0) coords, popularity comp = col 0
def _cosrows(Uh,E,rep,it,has):
    En=_unit(E); c=(Uh[rep]*En[it]).sum(1); c=np.where(has[rep],c,np.nan); c[np.linalg.norm(E[it],axis=1)<1e-6]=np.nan; return c.astype(np.float32)
def pp_feats(D,X,rows,groups=('cos','rec','prod','user','multi')):
    S=D.snap(X); e=rows['e']; ev=e[::10]; n=len(ev); it=rows['item']; rep=np.repeat(np.arange(n),10); out={}
    H0=hist_weights(D,ev,None); has=np.asarray(H0.sum(1)).ravel()>0
    def centroid(H,E): UV=np.asarray(H@_unit(E)); return UV, _unit(UV)
    U20=_space(D,S,20,'inv',1.0,16)
    E12=U20[:,:12]; E16=U20; E8=U20[:,:8]; E13=U20[:,1:13]
    UV,Uh=centroid(H0,E12)
    out['pp12_cos']=_cosrows(Uh,E12,rep,it,has); out['pp12_cos_z'],out['pp12_cos_dm']=_zwithin(out['pp12_cos'])
    for nm,E in (('pp8',E8),('pp16',E16),('pp12x',E13)):
        _,Uh_=centroid(H0,E); out[nm+'_cos']=_cosrows(Uh_,E,rep,it,has); out[nm+'_cos_z'],_=_zwithin(out[nm+'_cos'])
    out['pp12_unorm']=np.where(has,np.linalg.norm(UV,axis=1),np.nan)[rep].astype(np.float32)
    out['pp12_ucos_dot']=(out['pp12_cos']*out['pp12_unorm']).astype(np.float32)
    # un-whitened (p=0.5) variant
    if 'cos' in groups:
        M60=click_pairs_matrix(D,S,60,'one'); P60=ppmi(M60,1.0,0.0); U60,s60=svd_embed(P60,24,n_iter=7); E60=U60[:,:12]
        _,Uh_=centroid(H0,E60); out['ppb12_cos']=_cosrows(Uh_,E60,rep,it,has); out['ppb12_cos_z'],_=_zwithin(out['ppb12_cos'])
    if 'rec' in groups:
        for tag,tau,kind in (('e3',3.0,'exp'),('e10',10.0,'exp'),('p1',1.0,'pow')):
            Hr=hist_weights(D,ev,tau,kind); _,Uh_=centroid(Hr,E12); out[f'pp12_rec_{tag}']=_cosrows(Uh_,E12,rep,it,has)
        # old (first) end of history: reverse-recency, to test order information
        Hf=hist_weights(D,ev,None,maxlen=3); _,Uh_=centroid(Hf,E12); out['pp12_last3']=_cosrows(Uh_,E12,rep,it,has)
        # max-sim to last 5 history items (position decay)
        st=D.OFF[ev]; en=D.OFF[ev+1]; En=_unit(E12); best=np.full(n*10,-1.0,np.float32); wsum=np.zeros(n*10,np.float32)
        for j in range(5):
            ok=(en-st)>j; idx=np.where(ok,en-1-j,0); hv=En[D.HI[idx]]*ok[:,None]
            s_=(hv[rep]*En[it]).sum(1); best=np.where(ok[rep],np.maximum(best,s_),best); wsum+=np.where(ok[rep],s_*0.7**j,0)
        out['pp12_max5']=np.where(has[rep],best,np.nan).astype(np.float32); out['pp12_decay5']=np.where(has[rep],wsum,np.nan).astype(np.float32)
    if 'prod' in groups or 'user' in groups:
        En=_unit(E12)
        if 'prod' in groups:
            P=Uh[rep]*En[it]
            for d in range(12): out[f'pp_prod{d}']=np.where(has[rep],P[:,d],np.nan).astype(np.float32)
            for d in range(12): out[f'pp_item{d}']=En[it,d]
        if 'user' in groups:
            for d in range(12): out[f'pp_user{d}']=np.where(has[rep],Uh[rep,d],np.nan).astype(np.float32)
    if 'multi' in groups:
        # time-window fit: only clicks in the last 3 days before X (second 'time point'); and half-window (older) 
        for tag,lo in (('w3',X-3*DAY),):
            m=S['ct']>=lo; S2={k_:v[m] for k_,v in S.items() if k_ in ('cu','ci','ct')}
            U3=_space(D,S2,20,'inv',1.0,12); _,Uh_=centroid(H0,U3); out[f'pp12_{tag}_cos']=_cosrows(Uh_,U3,rep,it,has)
        out['pp12_w3_cos_z'],_=_zwithin(out['pp12_w3_cos'])
    return out
if __name__=='__main__':
    import time; c=int(sys.argv[1]); D=Data(); t=time.time()
    if c==27:
        V=extend_eval(D); nt=int(np.searchsorted(D.ETS,Jd(27))); assert nt==686824 and nt==D.NE-52939
    else: V=D.val(c)
    df=pl.DataFrame({k:np.asarray(v,np.float32) for k,v in pp_feats(D,Jd(c),V).items()}).with_columns(pl.Series('rowid',V['rowid']))
    import os; os.makedirs('/workspace/huawei/dw_p33/cache',exist_ok=True); df.write_parquet(f'/workspace/huawei/dw_p33/cache/pp_c{c}.parquet'); print(c,df.shape,round(time.time()-t),'s',flush=True)
