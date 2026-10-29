"""v17cooc: teammate(p33)-style item co-occurrence embeddings. python mkcooc.py VARIANT NAME...   VARIANT in A|B ; NAME in f1..f6,val23,val25,val26,eval
Graph = item x item symmetric co-occurrence over users' time-ordered, minute-deduped click logs (window W=20 click positions, weight 1/d; 'ppb' variant W=60 weight 1),
PPMI (context smoothing alpha=1.0, shift 0) -> randomized SVD (k=16+8 comps, n_iter 7), whitened coords (p=0): E16=cols0..15, E12=cols0..11, E8, E12x=cols1..12; ppb: un-whitened SVD(24)->12.
User vector = centroid of unit item vectors of own pre-ts history (uniform / exp tau 3,10 / 1/(1+pos) / last3 / max5 / decay0.7^j); features = cosines, mean sim, per-dim products, item+user coords, 3-day-window graph.
 A (train-only): click log = train events with ts<X (histories of those events + their clicked candidates, labels of ts<X only) -> exactly DW's leak-safe build.
 B (train+eval histories, TRANSDUCTIVE, Eric-approved for this feature only): A's click log UNION history_item_ids/history_timestamps (no labels) of the events in the target table
   (for NAME=eval: the 52,939 evaluation events; for fold/val tables: that table's events = exact mirror).  No clicked labels of any target-window event feed the graph.
Output: feat_{A,B}/NAME.parquet  rowid + co{A|B}_* float32."""
import sys, os, time; sys.dont_write_bytecode=True
sys.path.insert(0,'/workspace/huawei/dw_p33'); sys.path.insert(0,'/workspace/huawei/v12')
from p33lib import *          # also brings sigs: np, pl, Data, Jd, DAY, extend_eval, _zwithin via sigs
from sigs import _zwithin, extend_eval
from mkdw import rows_train, CUT
O='/workspace/huawei/v17cooc'
def _unit(a): return a/(np.linalg.norm(a,axis=-1,keepdims=True)+1e-9)
def _space(D,S,W=20,wm='inv',alpha=1.0,k=16,seed=0):
    P=ppmi(click_pairs_matrix(D,S,W,wm),alpha,0.0); U,s=svd_embed(P,k+8,n_iter=7,seed=seed); return U[:,:k]
def _cosrows(Uh,E,rep,it,has):
    En=_unit(E); c=(Uh[rep]*En[it]).sum(1); c=np.where(has[rep],c,np.nan); c[np.linalg.norm(E[it],axis=1)<1e-6]=np.nan; return c.astype(np.float32)
def extend_S(D,S,ev,n_orig_users):
    """union of S (dedup click log) with all history entries of events ev (labels not used). unseen users get a private pseudo-id per event."""
    st=D.OFF[ev]; ln=D.Hlen[ev]; tot=int(ln.sum())
    idx=np.repeat(st-np.concatenate([[0],np.cumsum(ln)[:-1]]),ln)+np.arange(tot)
    eu=D.EU[ev].astype(np.int64); eu=np.where(eu>=n_orig_users,D.NU+1+ev,eu)
    hu=np.repeat(eu,ln); hi=D.HI[idx].astype(np.int64); ht=D.HT[idx]
    cu=np.concatenate([S['cu'].astype(np.int64),hu]); ci=np.concatenate([S['ci'].astype(np.int64),hi]); ct=np.concatenate([S['ct'],ht])
    tm=(ct-Jd(1))//60; NIx=D.NI
    key=np.unique((cu*NIx+ci)*(1<<20)+tm); ui=key>>20; tm=key&((1<<20)-1)
    return dict(nX=S['nX'],cu=ui//NIx,ci=(ui%NIx).astype(np.int32),ct=tm*60+Jd(1)),int(len(key)-len(S['cu']))
def pp_feats(D,X,rows,variant,n_orig_users):
    S=D.snap(X); e=rows['e']; ev=e[::10]; n=len(ev); it=rows['item']; rep=np.repeat(np.arange(n),10); out={}
    nadd=0
    if variant=='B': S,nadd=extend_S(D,S,ev,n_orig_users)
    H0=hist_weights(D,ev,None); has=np.asarray(H0.sum(1)).ravel()>0
    hlen=np.asarray(H0.sum(1)).ravel()
    def centroid(H,E): UV=np.asarray(H@_unit(E)); return UV, _unit(UV)
    U20=_space(D,S,20,'inv',1.0,16); E12=U20[:,:12]; E16=U20; E8=U20[:,:8]; E13=U20[:,1:13]
    UV,Uh=centroid(H0,E12)
    out['pp12_cos']=_cosrows(Uh,E12,rep,it,has); out['pp12_cos_z'],out['pp12_cos_dm']=_zwithin(out['pp12_cos'])
    En12=_unit(E12)
    out['pp12_msim']=np.where(has[rep],(UV[rep]*En12[it]).sum(1)/np.maximum(hlen[rep],1),np.nan).astype(np.float32)   # mean cosine to history items
    out['pp12_dot']=np.where(has[rep],((np.asarray(H0@E12)[rep]/np.maximum(hlen[rep],1)[:,None])*E12[it]).sum(1),np.nan).astype(np.float32)  # raw (un-normalised) user-vector . item-vector
    for nm,E in (('pp8',E8),('pp16',E16),('pp12x',E13)):
        _,Uh_=centroid(H0,E); out[nm+'_cos']=_cosrows(Uh_,E,rep,it,has); out[nm+'_cos_z'],_=_zwithin(out[nm+'_cos'])
    out['pp12_unorm']=np.where(has,np.linalg.norm(UV,axis=1),np.nan)[rep].astype(np.float32)
    out['pp12_ucos_dot']=(out['pp12_cos']*out['pp12_unorm']).astype(np.float32)
    M60=click_pairs_matrix(D,S,60,'one'); P60=ppmi(M60,1.0,0.0); U60,s60=svd_embed(P60,24,n_iter=7); E60=U60[:,:12]
    _,Uh_=centroid(H0,E60); out['ppb12_cos']=_cosrows(Uh_,E60,rep,it,has); out['ppb12_cos_z'],_=_zwithin(out['ppb12_cos'])
    for tag,tau,kind in (('e3',3.0,'exp'),('e10',10.0,'exp'),('p1',1.0,'pow')):
        Hr=hist_weights(D,ev,tau,kind); _,Uh_=centroid(Hr,E12); out[f'pp12_rec_{tag}']=_cosrows(Uh_,E12,rep,it,has)
        if tag=='e3':
            Hn=Hr.multiply(1.0/np.maximum(np.asarray(Hr.sum(1)).ravel(),1e-9)[:,None]).tocsr()
            out['pp12_dotrec_e3']=np.where(has[rep],((np.asarray(Hn@E12)[rep])*E12[it]).sum(1),np.nan).astype(np.float32)
    Hf=hist_weights(D,ev,None,maxlen=3); _,Uh_=centroid(Hf,E12); out['pp12_last3']=_cosrows(Uh_,E12,rep,it,has)
    st=D.OFF[ev]; en=D.OFF[ev+1]; En=_unit(E12); best=np.full(n*10,-1.0,np.float32); wsum=np.zeros(n*10,np.float32)
    for j in range(5):
        ok=(en-st)>j; idx=np.where(ok,en-1-j,0); hv=En[D.HI[idx]]*ok[:,None]
        s_=(hv[rep]*En[it]).sum(1); best=np.where(ok[rep],np.maximum(best,s_),best); wsum+=np.where(ok[rep],s_*0.7**j,0)
    out['pp12_max5']=np.where(has[rep],best,np.nan).astype(np.float32); out['pp12_decay5']=np.where(has[rep],wsum,np.nan).astype(np.float32)
    P=Uh[rep]*En[it]
    for d in range(12): out[f'pp_prod{d}']=np.where(has[rep],P[:,d],np.nan).astype(np.float32)
    for d in range(12): out[f'pp_item{d}']=En[it,d]
    for d in range(12): out[f'pp_user{d}']=np.where(has[rep],Uh[rep,d],np.nan).astype(np.float32)
    lo=X-3*DAY; m=S['ct']>=lo; S2={k_:v[m] for k_,v in S.items() if k_ in ('cu','ci','ct')}
    U3=_space(D,S2,20,'inv',1.0,12); _,Uh_=centroid(H0,U3); out['pp12_w3_cos']=_cosrows(Uh_,U3,rep,it,has); out['pp12_w3_cos_z'],_=_zwithin(out['pp12_w3_cos'])
    return out,nadd,len(S['cu'])
if __name__=='__main__':
    var=sys.argv[1]; assert var in 'AB'; D=Data(); n_orig=len(D.uu)
    for nm in sys.argv[2:]:
        t=time.time(); X=CUT[nm]
        if nm=='eval':
            nt=D.NE; V=extend_eval(D); assert int(np.searchsorted(D.ETS,X))==nt
        else: V=rows_train(D,nm)
        o,nadd,nS=pp_feats(D,X,V,var,n_orig)
        df=pl.DataFrame({f'co{var}_{k}':np.asarray(v,np.float32) for k,v in o.items()}).with_columns(pl.Series('rowid',V['rowid'])).select(['rowid']+[f'co{var}_{k}' for k in o])
        df.write_parquet(f'{O}/feat_{var}/{nm}.parquet'); s=round(time.time()-t)
        msg=f'{var} {nm} shape={df.shape} clicklog={nS} added_by_eval_hist={nadd} {s}s'; print(msg,flush=True); open(f'{O}/logs/mk.log','a').write(msg+'\n')
