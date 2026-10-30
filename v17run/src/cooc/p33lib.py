"""dw_p33 reusable: history-co-occurrence PPMI + SVD embeddings and recency-weighted user vectors (teammate p33-style).
Leak-safe: item stats from train events ts<X only (D.snap(X)); per-row inputs = own pre-ts history. READ-ONLY w.r.t. other folders."""
import sys; sys.dont_write_bytecode=True
sys.path.insert(0,'@@WORK@@/dw'); sys.path.insert(0,'@@WORK@@/cooc')
from sigs import *
from scipy import sparse
from sklearn.utils.extmath import randomized_svd
from sklearn.metrics import roc_auc_score as AUC

def click_pairs_matrix(D,S,W=20,wmode='inv'):
    """symmetric item x item co-occurrence counts over the dedup click log S (users' clicks sorted by time), window W (distance in click order).
    wmode: 'one' -> weight 1 ; 'inv' -> 1/d ; 'sqrt' -> 1/sqrt(d)"""
    o=np.lexsort((S['ct'],S['cu'])); u=S['cu'][o]; it=S['ci'][o].astype(np.int64); NI=D.NI
    rows=[];cols=[];vals=[]
    for d in range(1,W+1):
        m=u[d:]==u[:-d]
        if not m.any(): break
        a,b=it[:-d][m],it[d:][m]; ok=a!=b; a,b=a[ok],b[ok]
        w=1.0 if wmode=='one' else (1.0/d if wmode=='inv' else d**-0.5)
        rows+= [a,b]; cols+=[b,a]; vals+=[np.full(len(a)*2,w,np.float32)]
    r=np.concatenate(rows); c=np.concatenate(cols); v=np.concatenate([np.full(len(x),1.0,np.float32) for x in rows]) if wmode=='one' else None
    # weights: rebuild aligned
    vals=[];
    for d in range(1,W+1):
        m=u[d:]==u[:-d]
        if not m.any(): break
        a,b=it[:-d][m],it[d:][m]; ok=a!=b; n=int(ok.sum()); w=1.0 if wmode=='one' else (1.0/d if wmode=='inv' else d**-0.5)
        vals+=[np.full(n,w,np.float32),np.full(n,w,np.float32)]
    # rows list is [a,b] per d (as row sides) -> the 'cols' list is [b,a]
    v=np.concatenate(vals)
    M=sparse.coo_matrix((v,(r,c)),shape=(NI,NI)).tocsr(); M.sum_duplicates(); return M

def ppmi(M,alpha=0.75,shift=0.0):
    """positive PMI with context-distribution smoothing alpha; shift=log(neg samples)"""
    M=M.tocoo(); r=np.asarray(M.sum(1)).ravel(); c=np.asarray(M.sum(0)).ravel(); T=M.data.sum()
    ca=c**alpha; ca=ca/ca.sum()*T
    pmi=np.log(M.data*T/(r[M.row]*ca[M.col]+1e-12))-shift; k=pmi>0
    return sparse.csr_matrix((pmi[k].astype(np.float32),(M.row[k],M.col[k])),shape=M.shape)

def svd_embed(P,k=256,n_iter=5,seed=0):
    U,s,Vt=randomized_svd(P,k,n_iter=n_iter,random_state=seed); return U.astype(np.float32),s.astype(np.float32)

def emb(U,s,k,p=0.5,norm_rows=False):
    E=U[:,:k]*(s[:k]**p); 
    if norm_rows: E=E/(np.linalg.norm(E,axis=1,keepdims=True)+1e-9)
    return E.astype(np.float32)

def hist_weights(D,e_idx,tau=None,kind='exp',maxlen=None):
    """csr (len(e_idx) x NI): own pre-ts history of events e_idx, weight by recency from the end (pos=0 newest). tau=None -> uniform.
    kind: 'exp' exp(-pos/tau), 'pow' 1/(1+pos)^tau"""
    st=D.OFF[e_idx]; en=D.OFF[e_idx+1]; ln=en-st
    ev=np.repeat(np.arange(len(e_idx)),ln); pos=np.repeat(en,ln)-1-np.arange(ln.sum())-0
    idx=np.concatenate([np.arange(a,b) for a,b in zip(st,en)]) if len(st)<2000 else (np.repeat(st-np.concatenate([[0],np.cumsum(ln)[:-1]]),ln)+np.arange(ln.sum()))
    pos=np.repeat(en,ln)-1-idx; it=D.HI[idx]
    if maxlen: m=pos<maxlen; ev,it,pos=ev[m],it[m],pos[m]
    if tau is None: w=np.ones(len(it),np.float32)
    elif kind=='exp': w=np.exp(-pos/tau).astype(np.float32)
    else: w=(1.0/(1+pos)**tau).astype(np.float32)
    H=sparse.csr_matrix((w,(ev,it)),shape=(len(e_idx),D.NI)); H.sum_duplicates(); return H
