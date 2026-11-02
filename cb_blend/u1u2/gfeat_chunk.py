"""Chunked re-implementation of v21graph/gfeat.feats (s_uv, s_h, cos_mean only) -- same math, low memory. Reads v21graph/out read-only."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ.setdefault(k,'4')
import numpy as np, sys; sys.dont_write_bytecode=True
O='/workspace/huawei/v21graph/out/'
def load_tab(nm): return dict(np.load(O+f'tab_{nm}.npz'))
def feats_chunk(T,U,V,ch=3000):
    NU=len(U); NI=len(V); user=T['user']; it=T['item']; H=T['hist']; E=len(H); n=len(it)
    u=np.minimum(user,NU-1); s_uv=(U[u]*V[it]).sum(1); s_uv[user>=NU]=0
    Vp=np.concatenate([V,np.zeros((1,V.shape[1]),np.float32)]); Vn=V/(np.linalg.norm(V,axis=1,keepdims=True)+1e-9)
    Vnp=np.concatenate([Vn,np.zeros((1,V.shape[1]),np.float32)])
    s_h=np.zeros(n,np.float32); cm=np.zeros(n,np.float32)
    for a in range(0,E,ch):
        b=min(E,a+ch); Hc=H[a:b]; m=(Hc!=NI).astype(np.float32); ne=m.sum(1)          # (e,30)
        hv=(Vp[Hc]*m[...,None]).sum(1)/np.maximum(ne,1)[:,None]                       # (e,d)
        itc=it[a*10:b*10].reshape(b-a,10)
        sh=np.einsum('ed,eqd->eq',hv,V[itc]); sh[ne==0]=0
        cs=np.einsum('ehd,eqd->eqh',Vnp[Hc],Vn[itc]); cs=cs*m[:,None,:]               # masked entries -> 0
        c=cs.sum(2)/np.maximum(ne,1)[:,None]; c[ne==0]=0
        s_h[a*10:b*10]=sh.ravel(); cm[a*10:b*10]=c.ravel()
    return dict(s_uv=s_uv.astype(np.float32),s_h=s_h,cos_mean=cm)
