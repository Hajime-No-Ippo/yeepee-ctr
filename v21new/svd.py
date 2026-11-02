import sys,time,numpy as np,scipy.sparse as sp; sys.dont_write_bytecode=True
from scipy.sparse.linalg import svds
from sklearn.metrics import roc_auc_score as AUC
G='/workspace/huawei/v21graph2/out/'; NI=50000; NU=53003
k=int(sys.argv[1]); mon=sys.argv[2]; K=int(sys.argv[3])
g=np.load(G+f'g_{k}.npz'); key=np.unique(g['cu'].astype(np.int64)*NI+g['ci'].astype(np.int64)); cu,ci=key//NI,key%NI
R=sp.csr_matrix((np.ones(len(cu),np.float32),(cu,ci)),shape=(NU,NI))
du=np.asarray(R.sum(1)).ravel(); di=np.asarray(R.sum(0)).ravel()
a=float(sys.argv[4]) if len(sys.argv)>4 else .5; b=float(sys.argv[5]) if len(sys.argv)>5 else .5
Rh=sp.diags(1/np.maximum(du,1)**a)@R@sp.diags(1/np.maximum(di,1)**b)
t=time.time(); U,S,Vt=svds(Rh,k=K); o=np.argsort(-S); S=S[o]; V=Vt[o].T; print('svd',time.time()-t,S[:5],flush=True)
T=np.load(G+f'tab_{mon}.npz'); y=T['y']; hist=T['hist'].astype(np.int64); us=np.minimum(T['user'].reshape(-1,10)[:,0].astype(np.int64),NU-1); item=T['item'].astype(np.int64).reshape(-1,10)
n=len(hist); X=np.concatenate([hist,np.full((n,0),NI)],1)
# fold-in set = hist ∪ user's graph items
Rr=R[us]; rows=sp.lil_matrix((n,NI+1),dtype=np.float32)
H=sp.csr_matrix((np.ones(hist.size,np.float32),(np.repeat(np.arange(n),hist.shape[1]),hist.ravel())),shape=(n,NI+1))[:,:NI]
M=((Rr+H)>0).astype(np.float32); cnt=np.asarray(M.sum(1)).ravel()
for beta in (.5,):
  P=(sp.diags(1/np.maximum(cnt,1)**a)@M@sp.diags(1/np.maximum(di,1)**b))@V   # n x K fold-in coords
  for gam in (0,1,2,3):
    Z=P*(S**gam)[None,:]
    sc=np.einsum('nk,nck->nc',Z,V[item])
    s=sc.reshape(-1); sd=(sc-sc.mean(1,keepdims=True)).reshape(-1)
    print(f'K{K} a{a} b{b} gam{gam} {mon} raw {AUC(y,s):.5f} dem {AUC(y,sd):.5f}',flush=True)
