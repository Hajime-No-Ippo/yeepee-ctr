"""Candidate signals. Each function: f(D, S, V, X) -> dict name->float32 array aligned to val rows (len n*10), using ONLY snapshot S (train events ts<X) + each row's own history/ts.
D = common.Data, S = D.snap(X), V = val dict (D.val(c)), X = fit cutoff (unix seconds)."""
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.linalg import svds
from common import *

def _imp_mats(D,X):
    nX=int(np.searchsorted(D.ETS,X)); u=np.repeat(D.EU[:nX],10); it=D.RI[:nX].ravel(); y=D.Y[:nX].ravel(); t=np.repeat(D.ETS[:nX],10)
    mk=lambda m,w=None: csr_matrix((np.ones(m.sum(),np.float32) if w is None else w[m],(u[m],it[m])),shape=(D.NU,D.NI))
    allm=np.ones(len(u),bool)
    return nX,mk(allm),mk(y==0),mk(y==1),mk((y==0)&(t>=X-3*DAY)),u,it,y,t

def hist_flat(D,V):
    """own-history items/ts of each val event, flattened, with row->event pointer"""
    e=V['e'][::10]; ln=D.Hlen[e]; off=np.zeros(len(e)+1,np.int64); off[1:]=np.cumsum(ln)
    idx=np.concatenate([np.arange(D.OFF[x],D.OFF[x+1]) for x in e]) if len(e) else np.array([],int)
    return D.HI[idx],D.HT[idx],off,ln

def sig_impression_svd(D,S,V,X,k=32,Mimp=None):
    """A. taste embedding from IMPRESSIONS (who was shown what), not clicks: SVD of user x item impression-count matrix."""
    nX,Mi,Ms,Mc,Mr,*_=_imp_mats(D,X)
    nu=np.asarray(Mi.sum(1)).ravel(); ni=np.asarray(Mi.sum(0)).ravel()
    r,c=Mi.nonzero(); w=np.asarray(Mi[r,c]).ravel()
    M=csr_matrix((w/np.sqrt((nu[r]+1)*(ni[c]+1)),(r,c)),shape=Mi.shape,dtype=np.float32)
    U,sv,Vt=svds(M,k=k,random_state=0); Vi=(Vt.T*sv).astype(np.float32); Us=(U*sv).astype(np.float32)
    u=V['user']; it=V['item']; has=(nu[u]>0)
    rec=np.einsum('ij,ij->i',U[u]*np.sqrt(sv),Vt.T[it]*np.sqrt(sv)); 
    cos=np.einsum('ij,ij->i',norm(Us[u]),norm(Vi[it]))
    out=dict(impsvd_rec=rec,impsvd_cos=cos); 
    for k_,v_ in out.items(): v_[~has]=np.nan
    z=within_event(cos); out['impsvd_cos_z']=z[0]; out['impsvd_cos_rk']=z[2]
    return {k_:v_.astype(np.float32) for k_,v_ in out.items()},dict(Vi=Vi)

def sig_skip_embedding(D,S,V,X,Vclk):
    """B. negative feedback in embedding space: centroid of items the user was SHOWN but did NOT click (all train before X / last 3d) and centroid of all shown items, cos with candidate (click-SVD item vectors),
    plus own-history centroid cos (reference) and skip-minus-history contrasts."""
    nX,Mi,Ms,Mc,Mr,*_=_imp_mats(D,X)
    u=V['user']; it=V['item']; Ec=Vclk[it]
    def cen(M):
        s=M@Vclk; n=np.asarray(M.sum(1)).ravel(); return s, n
    out={}
    for nm,M in (('skall',Ms),('skrec',Mr),('impall',Mi)):
        s,n=cen(M); cs=norm(s)[u]; c=np.einsum('ij,ij->i',cs,Ec).astype(np.float32); c[n[u]==0]=np.nan; out['c_'+nm]=c
    hi,ht,off,ln=hist_flat(D,V)
    hs=np.add.reduceat(np.vstack([Vclk[hi],np.zeros((1,Vclk.shape[1]),np.float32)]),np.minimum(off[:-1],len(hi)),axis=0) if len(hi) else None
    hs=np.where((ln>0)[:,None],hs,0); hc=np.repeat(norm(hs),10,axis=0)
    c=np.einsum('ij,ij->i',hc,Ec).astype(np.float32); c[np.repeat(ln==0,10)]=np.nan; out['c_hist']=c
    out['c_hist_minus_sk']=out['c_hist']-out['c_skall']; out['c_hist_minus_skrec']=out['c_hist']-out['c_skrec']
    # skip-centroid of the user's LAST event slate only (non-clicked), what the user just rejected
    return out,{}

def sig_slate_centrality(D,S,V,X,Vemb,tag='clk'):
    """C. slate coherence: candidate's mean / max cosine to the other 9 candidates in the same slate (odd-one-out / retrieved-cluster membership)."""
    E=Vemb[V['item']].reshape(-1,10,Vemb.shape[1]); G=np.einsum('nik,njk->nij',E,E); idx=np.arange(10); G[:,idx,idx]=np.nan
    mean=np.nanmean(G,2); mx=np.nanmax(G,2); top3=np.sort(np.nan_to_num(G,nan=-9),2)[:,:,-3:].mean(2)
    out={f'cent_{tag}_mean':mean.ravel(),f'cent_{tag}_max':mx.ravel(),f'cent_{tag}_top3':top3.ravel()}
    out[f'cent_{tag}_mean_dm']=(mean-mean.mean(1,keepdims=True)).ravel()
    out[f'slate_{tag}_coh']=np.repeat(mean.mean(1),10)
    return {k:v.astype(np.float32) for k,v in out.items()},{}

def sig_exposure_lift(D,S,V,X):
    """E. Pearson-residual of user-item exposure vs independence (u_imp * i_imp / N); same for skipped-only impressions."""
    nX,Mi,Ms,Mc,Mr,*_=_imp_mats(D,X); N=Mi.sum()
    u=V['user']; it=V['item']; ui=np.asarray(Mi[u,it]).ravel(); us=np.asarray(Ms[u,it]).ravel()
    nu=np.asarray(Mi.sum(1)).ravel()[u]; ni=np.asarray(Mi.sum(0)).ravel()[it]
    ex=nu*ni/N; out={'expo_resid':(ui-ex)/np.sqrt(ex+0.05),'expo_ratio':np.log((ui+0.5)/(ex+0.5)),'skip_cnt':us.astype(np.float32),'skip_frac':(us+0.0)/(ui+1)}
    return {k:v.astype(np.float32) for k,v in out.items()},{}

def sig_coimpression(D,S,V,X):
    """F. slate co-membership: for candidate c and each own-history item h, how often h and c were shown in the same train slate (before X), as lift over independence; sum/mean/max over h."""
    nX=int(np.searchsorted(D.ETS,X)); R=D.RI[:nX]
    B=csr_matrix((np.ones(nX*10,np.float32),(np.repeat(np.arange(nX),10),R.ravel())),shape=(nX,D.NI))
    C=(B.T@B).tocsr(); C.setdiag(0); C.eliminate_zeros(); ni=np.asarray(B.sum(0)).ravel()
    keys=None
    hi,ht,off,ln=hist_flat(D,V); n=len(V['e'])//10
    evi=np.repeat(np.arange(n),ln)                  # event of each history entry
    # all (hist item, candidate) pairs: for each event, 20x10
    pe=np.repeat(evi,10); ph=np.repeat(hi,10); pc=V['item'].reshape(n,10)[evi].ravel(); prow=(np.repeat(evi,10)*10+np.tile(np.arange(10),len(evi)))
    cnt=np.asarray(C[ph,pc]).ravel().astype(np.float32)
    Nimp=nX*10.0; ex=ni[ph]*ni[pc]*9.0/Nimp         # P(two given impressions share a slate)=9/N_imp
    lift=(cnt-ex)/np.sqrt(ex+0.05)
    out={}
    for nm,val in (('sum',cnt),('lift',lift)):
        s=np.bincount(prow,weights=val,minlength=n*10); out['coimp_'+nm+'_sum']=s
        out['coimp_'+nm+'_mean']=s/np.maximum(np.repeat(ln,10),1)
    mx=np.full(n*10,-1e9,np.float32); np.maximum.at(mx,prow,lift); mx[mx<-1e8]=np.nan; out['coimp_lift_max']=mx
    return {k:np.asarray(v,np.float32) for k,v in out.items()},{}

def sig_user_time(D,S,V,X):
    """H+D. user-level time/behaviour: hour-of-day match of click times / event times, last-event outcome, recent event click rate, same-time-bucket event click rate."""
    nX=int(np.searchsorted(D.ETS,X)); u=V['user']; t=V['ts']; n=len(u)//10
    hr=lambda ts: ((ts%DAY)//3600).astype(np.int64)
    # click hours per user from click log
    Hc=np.zeros((D.NU,24),np.float32); np.add.at(Hc,(S['cu'],hr(S['ct'])),1)
    He=np.zeros((D.NU,24),np.float32); np.add.at(He,(D.EU[:nX],hr(D.ETS[:nX])),1)
    h=hr(t)
    def win(H):
        w=sum(H[u,(h+d)%24] for d in (-2,-1,0,1,2)); tot=H[u].sum(1); return ((w+1)/(tot+24/5.0)/(5/24)).astype(np.float32),tot
    out={}; out['hr_click_match'],tc=win(Hc); out['hr_ev_match'],te=win(He)
    # last-event outcome / recent click rate / same-bucket ctr (event-level)
    evclk=D.Y[:nX].sum(1); eu=D.EU[:nX]; ets=D.ETS[:nX]; tb=((ets%DAY)//21600)
    order=np.lexsort((ets,eu)); eus=eu[order]; 
    last_e=np.full(D.NU,-1,np.int64)
    # vectorised: last event index per user = max index among that user's events
    np.maximum.at(last_e,eu,np.arange(nX))
    lc=np.where(last_e[u]>=0,evclk[np.maximum(last_e[u],0)],np.nan); out['last_ev_clicked']=lc.astype(np.float32)
    # recent-5 event click rate
    rank_in_user=np.empty(nX,np.int64); rank_in_user[order]=np.arange(nX)-np.searchsorted(eus,eus)
    cnt=np.bincount(eu,minlength=D.NU); frm_end=cnt[eu]-1-rank_in_user   # 0 = last
    m5=frm_end<5; s5=np.bincount(eu[m5],weights=evclk[m5],minlength=D.NU); n5=np.bincount(eu[m5],minlength=D.NU)
    out['ev_ctr_last5']=((s5[u]+0.629*2)/(n5[u]+2)).astype(np.float32)
    Sb=np.zeros((D.NU,4),np.float32); Nb=np.zeros((D.NU,4),np.float32); np.add.at(Sb,(eu,tb),evclk); np.add.at(Nb,(eu,tb),1)
    tbv=(h//6); out['ev_ctr_samebucket']=((Sb[u,tbv]+0.629*3)/(Nb[u,tbv]+3)).astype(np.float32)
    out['ev_cnt_samebucket_frac']=((Nb[u,tbv]+0.5)/(Nb[u].sum(1)+2)).astype(np.float32)
    return out,{}

def _svd_scaled(M,k):
    nu=np.asarray(M.sum(1)).ravel(); ni=np.asarray(M.sum(0)).ravel(); M=M.tocoo()
    Ms=csr_matrix((M.data/np.sqrt((nu[M.row]+1)*(ni[M.col]+1)),(M.row,M.col)),shape=M.shape,dtype=np.float32)
    U,sv,Vt=svds(Ms,k=k,random_state=0); return (U*sv).astype(np.float32),(Vt.T*sv).astype(np.float32),nu

def sig_impression_variants(D,S,V,X):
    """A2. variants of the impression-taste embedding: other ranks, skip-only SVD, joint click+impression SVD, recent-impression centroid, and the user's click-history centroid measured in IMPRESSION item space."""
    nX,Mi,Ms,Mc,Mr,*_=_imp_mats(D,X); u=V['user']; it=V['item']; out={}
    def ucos(U_,V_,nu,tag):
        c=np.einsum('ij,ij->i',norm(U_[u]),norm(V_[it])).astype(np.float32); c[nu[u]==0]=np.nan; out[tag]=c; out[tag+'_z']=within_event(c)[0]
    for k in (16,64):
        U_,V_,nu=_svd_scaled(Mi,k); ucos(U_,V_,nu,f'impsvd{k}_cos')
    U_,V_,nu=_svd_scaled(Ms,32); ucos(U_,V_,nu,'skipsvd32_cos')
    # joint click(log, dedup)+impression
    ck=csr_matrix((np.ones(len(S['cu']),np.float32),(S['cu'],S['ci'])),shape=(D.NU,D.NI)); ck.data[:]=1
    from scipy.sparse import hstack
    U_,V_,nu=_svd_scaled(hstack([ck*3.0,Mi]).tocsr(),32); Vj=V_[:D.NI]  # click-part item vecs, joint user space
    c=np.einsum('ij,ij->i',norm(U_[u]),norm(Vj[it])).astype(np.float32); out['joint32_cos_clkpart']=c; out['joint32_cos_clkpart_z']=within_event(c)[0]
    Vimp=_svd_scaled(Mi,32)[1]; Vn=norm(Vimp)
    # own-history centroid in impression item space
    hi,ht,off,ln=hist_flat(D,V); n=len(ln)
    hs=np.add.reduceat(np.vstack([Vn[hi],np.zeros((1,32),np.float32)]),np.minimum(off[:-1],len(hi)),axis=0); hs=np.where((ln>0)[:,None],hs,0)
    c=np.einsum('ij,ij->i',np.repeat(norm(hs),10,axis=0),Vn[it]).astype(np.float32); c[np.repeat(ln==0,10)]=np.nan; out['hist_cos_impspace']=c; out['hist_cos_impspace_z']=within_event(c)[0]
    # recent (3d) impression centroid in click space needs Vclk -> in compute.py
    return out,{}

def sig_recent_impression(D,S,V,X,Vclk,days=(1,3)):
    nX=int(np.searchsorted(D.ETS,X)); u=V['user']; it=V['item']; out={}
    for d in days:
        m=D.ETS[:nX]>=X-d*DAY; r=np.nonzero(m)[0]
        M=csr_matrix((np.ones(len(r)*10,np.float32),(np.repeat(D.EU[r],10),D.RI[r].ravel())),shape=(D.NU,D.NI)); n=np.asarray(M.sum(1)).ravel()
        c=np.einsum('ij,ij->i',norm(M@Vclk)[u],Vclk[it]).astype(np.float32); c[n[u]==0]=np.nan; out[f'c_imp{d}d']=c
        # decayed by event: also impression count in window of this exact item
        out[f'ui_imp{d}d']=np.asarray(M[u,it]).ravel().astype(np.float32)
    return out,{}

def sig_impression_multirank(D,S,V,X,ks=(8,64,128)):
    """A3. impression-SVD at more ranks: user-cos, plus own-click-history centroid cos and max-sim (over history items) in that impression item space."""
    nX,Mi,*_=_imp_mats(D,X); u=V['user']; it=V['item']; out={}
    hi,ht,off,ln=hist_flat(D,V); n=len(ln); evi=np.repeat(np.arange(n),ln); itm=it.reshape(n,10)
    for k in ks:
        U_,V_,nu=_svd_scaled(Mi,k); Vn=norm(V_)
        if k in (8,128): c=np.einsum('ij,ij->i',norm(U_[u]),Vn[it]).astype(np.float32); c[nu[u]==0]=np.nan; out[f'impsvd{k}_cos']=c
        hs=np.zeros((n,k),np.float32); np.add.at(hs,evi,Vn[hi]); c=np.einsum('ij,ij->i',np.repeat(norm(hs),10,axis=0),Vn[it]).astype(np.float32); c[np.repeat(ln==0,10)]=np.nan; out[f'hist_cos_imp{k}']=c
        if k==64:   # max-sim over history items
            mx=np.full((n,10),-1.0,np.float32); H=Vn[hi]
            for j in range(10):
                s=np.einsum('ik,ik->i',H,Vn[itm[evi,j]]); np.maximum.at(mx[:,j],evi,s)
            mx=mx.ravel(); mx[np.repeat(ln==0,10)]=np.nan; out['hist_maxsim_imp64']=mx
    for kk in list(out): out[kk+'_z']=within_event(np.nan_to_num(out[kk],nan=np.nanmedian(out[kk])))[0]
    return out,{}
