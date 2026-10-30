"""v19 NMF / LDA topic features. python mkmf.py VARIANT NAME...   VARIANT in A|B ; NAME in f1..f6,val23,val25,val26,eval
Graph variants are IDENTICAL to v17cooc (mkcooc.py): A = train-only click log (D.snap(X)); B = A U history entries (no labels) of the target-table events (TRANSDUCTIVE, Eric-approved).
Three k=12 non-negative spaces (item topic matrix T, NIxk):
  nmfU : NMF(12) of the user x item click matrix (weights 1/sqrt((nu+1)(ni+1)), as DW's matrix); T = H^T (item loadings)
  nmfP : NMF(12) of the item x item PPMI matrix (W=20, 1/d, alpha=1) ; T = W
  ldaU : LDA(12) (batch, 15 it) on user x item binary click matrix; T[i,t] = p(i|t)/sum_t p(i|t)
Per space: user profile = mean over the row's OWN pre-ts history of T_i (L1-normalised) (uniform / exp tau3,10 / 1/(1+pos) / last3 / max5 / 0.7^j); features = cos(user,unit item), dot (match prob), lift (dot / global topic prior),
 within-event z/demeaned cos, user & item topic entropy; nmfU also: per-dim products (12), and the fitted user factor's reconstruction W_u.H_i (only for users present in the graph).
Output feat_N{A|B}/NAME.parquet  rowid + nm{A|B}_* float32"""
import sys, os, time; sys.dont_write_bytecode=True
sys.path.insert(0,'/workspace/huawei/v17cooc')
from mkcooc import *
from mkcooc import _zwithin
from scipy import sparse
from sklearn.decomposition import NMF, LatentDirichletAllocation
O='/workspace/huawei/v19'; K=12
def _unit(a): return a/(np.linalg.norm(a,axis=-1,keepdims=True)+1e-9)
def nmf(M,seed=0):
    m=NMF(K,init='nndsvda',solver='mu',max_iter=150,tol=1e-4,random_state=seed); W=m.fit_transform(M); return W.astype(np.float32),m.components_.T.astype(np.float32),m.n_iter_
def spaces(D,S,uq,ui):
    M=sparse.csr_matrix((np.ones(len(ui),np.float32),(ui,S['ci'].astype(np.int64))),shape=(len(uq),D.NI)); M.sum_duplicates(); M.data[:]=1
    nu=np.asarray(M.sum(1)).ravel(); ni=np.asarray(M.sum(0)).ravel(); Mc=M.tocoo()
    Mw=sparse.csr_matrix((1/np.sqrt((nu[Mc.row]+1)*(ni[Mc.col]+1)),(Mc.row,Mc.col)),shape=M.shape,dtype=np.float32)
    Wu,Hu,it1=nmf(Mw)
    P=ppmi(click_pairs_matrix(D,S,20,'inv'),1.0,0.0); Wp,_,it2=nmf(P)
    lda=LatentDirichletAllocation(K,learning_method='batch',max_iter=15,random_state=0,n_jobs=8,evaluate_every=-1); lda.fit(M)
    C=lda.components_; T=(C/C.sum(1,keepdims=True)).T.astype(np.float32)
    return dict(nmfU=Hu,nmfP=Wp,ldaU=T),Wu,(it1,it2)
def space_feats(tag,T,D,ev,rep,it,has,hlen,out):
    nz=T.sum(1)>1e-9; tI=T/np.maximum(T.sum(1,keepdims=True),1e-12); uI=_unit(T); prior=tI[nz].mean(0)+1e-9
    def prof(H):   # H csr (n x NI) -> user L1 profile (n x K), unit profile
        P=np.asarray(H@tI); n1=P.sum(1,keepdims=True); return P/np.maximum(n1,1e-12)
    def cos_of(H):
        UV=np.asarray(H@uI); Uh=_unit(UV); c=(Uh[rep]*uI[it]).sum(1); c=np.where(has[rep]&nz[it],c,np.nan); return c.astype(np.float32),Uh
    H0=hist_weights(D,ev,None)
    c,Uh=cos_of(H0); out[f'{tag}_cos']=c; out[f'{tag}_cos_z'],out[f'{tag}_cos_dm']=_zwithin(c)
    pu=prof(H0); aff=np.where(has[rep]&nz[it],(pu[rep]*tI[it]).sum(1),np.nan).astype(np.float32); out[f'{tag}_aff']=aff
    out[f'{tag}_aff_z'],_=_zwithin(aff)
    lift=np.where(has[rep]&nz[it],(pu[rep]/prior*tI[it]).sum(1),np.nan).astype(np.float32); out[f'{tag}_lift']=lift
    ent=lambda p:-(p*np.log(p+1e-9)).sum(1)
    out[f'{tag}_uent']=np.where(has,ent(pu),np.nan)[rep].astype(np.float32); out[f'{tag}_ient']=np.where(nz,ent(tI),np.nan)[it].astype(np.float32)
    for nm,(tau,kind) in (('e3',(3.0,'exp')),('e10',(10.0,'exp')),('p1',(1.0,'pow'))):
        Hr=hist_weights(D,ev,tau,kind); c2,_=cos_of(Hr); out[f'{tag}_cos_{nm}']=c2
        if nm=='e3': pr=prof(Hr); out[f'{tag}_aff_e3']=np.where(has[rep]&nz[it],(pr[rep]*tI[it]).sum(1),np.nan).astype(np.float32)
    Hf=hist_weights(D,ev,None,maxlen=3); out[f'{tag}_last3'],_=cos_of(Hf)
    st=D.OFF[ev]; en=D.OFF[ev+1]; n=len(ev); best=np.full(n*10,-1.0,np.float32); wsum=np.zeros(n*10,np.float32)
    for j in range(5):
        ok=(en-st)>j; idx=np.where(ok,en-1-j,0); hv=uI[D.HI[idx]]*ok[:,None]
        s_=(hv[rep]*uI[it]).sum(1); best=np.where(ok[rep],np.maximum(best,s_),best); wsum+=np.where(ok[rep],s_*0.7**j,0)
    out[f'{tag}_max5']=np.where(has[rep]&nz[it],best,np.nan).astype(np.float32); out[f'{tag}_decay5']=np.where(has[rep]&nz[it],wsum,np.nan).astype(np.float32)
    if tag=='nmfU':
        Pd=Uh[rep]*uI[it]
        for d in range(K): out[f'{tag}_prod{d}']=np.where(has[rep]&nz[it],Pd[:,d],np.nan).astype(np.float32)
def mf_feats(D,X,rows,variant,n_orig):
    S=D.snap(X); e=rows['e']; ev=e[::10]; n=len(ev); it=rows['item']; rep=np.repeat(np.arange(n),10); out={}; nadd=0
    user=rows['user'].astype(np.int64)[::10]
    if variant=='B':
        S,nadd=extend_S(D,S,ev,n_orig); user=np.where(user>=n_orig,D.NU+1+ev,user)   # same pseudo-id rule as extend_S
    uq,ui=np.unique(S['cu'].astype(np.int64),return_inverse=True)
    T,Wu,its=spaces(D,S,uq,ui)
    H0=hist_weights(D,ev,None); has=np.asarray(H0.sum(1)).ravel()>0; hlen=np.asarray(H0.sum(1)).ravel()
    for tag,Tm in T.items(): space_feats(tag,Tm,D,ev,rep,it,has,hlen,out)
    # fitted-user-factor reconstruction (only users present in the graph)
    pos=np.minimum(np.searchsorted(uq,user),len(uq)-1); inl=uq[pos]==user
    r=(Wu[pos][rep]*T['nmfU'][it]).sum(1); r=np.where(inl[rep],r,np.nan).astype(np.float32); out['nmfU_wu_rec']=r; out['nmfU_wu_rec_z'],_=_zwithin(r)
    return out,nadd,len(S['cu']),its
if __name__=='__main__':
    var=sys.argv[1]; assert var in 'AB'; D=Data(); n_orig=len(D.uu)
    for nm in sys.argv[2:]:
        t=time.time(); X=CUT[nm]
        if nm=='eval':
            nt=D.NE; V=extend_eval(D); assert int(np.searchsorted(D.ETS,X))==nt
        else: V=rows_train(D,nm)
        o,nadd,nS,its=mf_feats(D,X,V,var,n_orig)
        df=pl.DataFrame({f'nm{var}_{k}':np.asarray(v,np.float32) for k,v in o.items()}).with_columns(pl.Series('rowid',V['rowid'])).select(['rowid']+[f'nm{var}_{k}' for k in o])
        df.write_parquet(f'{O}/feat_N{var}/{nm}.parquet'); s=round(time.time()-t)
        msg=f'{var} {nm} shape={df.shape} clicklog={nS} added_by_eval_hist={nadd} nmf_iters={its} {s}s'; print(msg,flush=True); open(f'{O}/logs/mk.log','a').write(msg+'\n')
