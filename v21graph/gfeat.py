import numpy as np,sys; sys.dont_write_bytecode=True
O='/workspace/huawei/v21graph/out/'
def load_tab(nm): return np.load(O+f'tab_{nm}.npz')
def feats(nm,U,V):
    """graph features for table nm from propagated embeddings U (NU,d), V (NI,d). returns dict of (n_rows,) arrays"""
    T=load_tab(nm); NU=len(U); NI=len(V); u=np.minimum(T['user'],NU-1); it=T['item']; H=T['hist']; n=len(it)
    s_uv=(U[u]*V[it]).sum(1); s_uv[T['user']>=NU]=0
    Vp=np.concatenate([V,np.zeros((1,V.shape[1]),np.float32)]); m=(H!=NI).astype(np.float32); ne=m.sum(1)
    hv=(Vp[H]*m[...,None]).sum(1)/np.maximum(ne,1)[:,None]   # mean hist item emb (n_events,d)
    rep=np.repeat(np.arange(len(H)),10)
    s_h=(hv[rep]*V[it]).sum(1); s_h[ne[rep]==0]=0
    Vn=V/(np.linalg.norm(V,axis=1,keepdims=True)+1e-9); Vnp=np.concatenate([Vn,np.zeros((1,V.shape[1]),np.float32)])
    cs=np.einsum('rhd,rd->rh',Vnp[H[rep]],Vn[it]); cs=np.where(m[rep]>0,cs,-9); mx=cs.max(1); mx[ne[rep]==0]=0
    cm=np.where(m[rep]>0,cs,0).sum(1)/np.maximum(ne[rep],1)
    return dict(s_uv=s_uv,s_h=s_h,cos_max=mx,cos_mean=cm,y=T['y'],hlen=ne[rep])
