"""item-item PPR-style scores. python ppr.py K TABLE -> out/ppr_{TABLE}.npy (n_rows, nfeat)"""
import sys,numpy as np,scipy.sparse as sp,time; sys.dont_write_bytecode=True
k=int(sys.argv[1]); nm=sys.argv[2]; O='/workspace/huawei/v21graph/out/'
NU,NI=np.load(O+'dims.npy'); g=np.load(O+f'g_{k}.npz'); key=np.unique(g['cu'].astype(np.int64)*NI+g['ci'].astype(np.int64)); eu=key//NI; ei=key%NI
B=sp.csr_matrix((np.ones(len(key),np.float32),(eu,ei)),shape=(NU,NI))
du=np.asarray(B.sum(1)).ravel(); di=np.asarray(B.sum(0)).ravel()
Bu=sp.diags(1/np.maximum(du,1)**.5)@B           # user-degree damped
C=(Bu.T@Bu).tocsr(); C.setdiag(0); C.eliminate_zeros()
S=sp.diags(1/np.maximum(di,1)**.5)@C@sp.diags(1/np.maximum(di,1)**.5); S=S.tocsr().astype(np.float32)
print('S nnz',S.nnz,flush=True)
T=np.load(O+f'tab_{nm}.npz'); H=T['hist']; it=T['item'].reshape(-1,10) if False else T['item']; n=len(H); t0=time.time()
rep=np.repeat(np.arange(n),10); out=np.zeros((len(it),4),np.float32)   # hop1,hop2,ppr(.5 mix,3 hops), hop1 w/ recency
for s in range(0,n,1500):
    e=np.arange(s,min(n,s+1500)); h=H[e]; m=h!=NI; r=np.repeat(np.arange(len(e)),30)[m.ravel()]; c=h[m]
    x0=np.zeros((len(e),NI),np.float32); np.add.at(x0,(r,c),1.0)
    x1=(S@x0.T).T; x2=(S@x1.T).T; x3=(S@x2.T).T
    ppr=.5*x0*0+.5*x1+.25*x2+.125*x3
    rows=np.arange(s*10,min(n,s+1500)*10); cr=np.repeat(np.arange(len(e)),10); ci_=it[rows]
    out[rows,0]=x1[cr,ci_]; out[rows,1]=x2[cr,ci_]; out[rows,2]=ppr[cr,ci_]
    # recency-weighted hop1
    w=np.zeros((len(e),NI),np.float32); wt=(0.9**np.arange(30))[None,:].repeat(len(e),0)[m]; np.add.at(w,(r,c),wt); out[rows,3]=(S@w.T).T[cr,ci_]
    if s%6000==0: print(s,time.time()-t0,flush=True)
np.save(O+f'ppr_{nm}.npy',out)
