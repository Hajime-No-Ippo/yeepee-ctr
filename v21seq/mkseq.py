"""mkseq.py VARIANT NAME...  (A|B ; NAME f1..f6,val23,val25,val26,eval)
Per-table sequence inputs for v21seq (copy-based; imports read-only from v17cooc/mkcooc.py, v12/mkdw.py):
 hidx (E,20) int32 history item idx, newest first, pad=NI ; hgap (E,20) float32 seconds (event ts - click ts), pad=-1 ; hlen
 cand (N,) int32 ; rowid (N,) ; eid index
 cos1 (N,20) fp16: cosine(candidate, hist item) in train-only PPMI-SVD space (W20 inv, whitened, 16 comps) at the table's cutoff X
 cos2 (N,20) fp16: same in W60-one SVD space (12 comps un-whitened)
 init32 (NI,32) fp16 (only saved with --init): whitened SVD coords, used to initialise ID embeddings.
A: graph = train events ts<X (labels ts<X only).  B (TRANSDUCTIVE): + the table's own events' history entries (no labels), exactly as v17cooc."""
import sys, os, time; sys.dont_write_bytecode=True
sys.path.insert(0,'/workspace/huawei/v17cooc'); 
from mkcooc import *
from mkcooc import _space,_unit
O='/workspace/huawei/v21seq/seqfeat'
def build(D,nm,var,n_orig,init=False):
    X=CUT[nm]
    if nm=='eval': nt=D.NE; V=extend_eval(D); assert int(np.searchsorted(D.ETS,X))==nt
    else: V=rows_train(D,nm)
    S=D.snap(X); ev=V['e'][::10]; n=len(ev)
    if var=='B': S,nadd=extend_S(D,S,ev,n_orig)
    U32=_space(D,S,20,'inv',1.0,32); E1=_unit(U32[:,:16])
    M60=click_pairs_matrix(D,S,60,'one'); P60=ppmi(M60,1.0,0.0); U60,s60=svd_embed(P60,24,n_iter=7); E2=_unit(U60[:,:12])
    st=D.OFF[ev]; ln=D.Hlen[ev].astype(np.int64); assert ln.max()<=20
    j=np.arange(20)[None]; ok=j<ln[:,None]; idx=np.where(ok,(D.OFF[ev+1][:,None]-1-j),0)
    hidx=np.where(ok,D.HI[idx],D.NI).astype(np.int32); hgap=np.where(ok,D.ETS[ev][:,None]-D.HT[idx],-1).astype(np.float32)
    cand=V['item'].astype(np.int32); rep=np.repeat(np.arange(n),10)
    hh=hidx[rep]; okr=ok[rep]; hs=np.where(okr,hh,0)
    c1=np.einsum('nd,nkd->nk',E1[cand],E1[hs]); c2=np.einsum('nd,nkd->nk',E2[cand],E2[hs])
    c1[~okr]=0; c2[~okr]=0
    c1[np.linalg.norm(U32[cand,:16],axis=1)<1e-6]=0; c2[np.linalg.norm(U60[cand,:12],axis=1)<1e-6]=0
    np.savez(f'{O}/{var}_{nm}.npz',hidx=hidx,hgap=hgap,hlen=ln.astype(np.int16),cand=cand,rowid=V['rowid'],ets=D.ETS[ev],
             cos1=c1.astype(np.float16),cos2=c2.astype(np.float16),NI=D.NI)
    if init: np.save(f'{O}/init_{var}_{nm}.npy',U32.astype(np.float16))
    return n
if __name__=='__main__':
    var=sys.argv[1]; assert var in 'AB'; init=os.environ.get('INIT')=='1'; D=Data(); n_orig=len(D.uu)
    for nm in sys.argv[2:]:
        t=time.time(); n=build(D,nm,var,n_orig,init=init); msg=f'{var} {nm} events={n} NI={D.NI} {time.time()-t:.0f}s'; print(msg,flush=True); open('/workspace/huawei/v21seq/logs/mkseq.log','a').write(msg+'\n')
