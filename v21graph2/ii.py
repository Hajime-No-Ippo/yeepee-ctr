"""item-item co-click graph (train-only, cutoff graph g_k): sessions = clicks by same user within window W seconds.
score(row)= aggregate over user's last-30 hist items h of Cn[h,item] (cosine-normalised co-count), several aggregations."""
import sys,numpy as np,scipy.sparse as sp; sys.dont_write_bytecode=True
O='/workspace/huawei/v21graph2/out/'; NU,NI=np.load(O+'dims.npy')
def build(k,W,dmax=40,tau=None):
    g=np.load(O+f'g_{k}.npz'); u,i,t=g['cu'].astype(np.int64),g['ci'].astype(np.int64),g['ct']
    o=np.lexsort((t,u)); u,i,t=u[o],i[o],t[o]; R=[];Cc=[];V=[]
    for d in range(1,dmax+1):
        a=slice(0,len(u)-d); b=slice(d,len(u)); m=(u[a]==u[b])&(i[a]!=i[b])&((t[b]-t[a])<=W)
        if not m.any(): break
        w=np.ones(m.sum(),np.float32) if tau is None else np.exp(-(t[b][m]-t[a][m])/tau).astype(np.float32)
        R.append(i[a][m]);Cc.append(i[b][m]);V.append(w)
    R=np.concatenate(R);Cc=np.concatenate(Cc);V=np.concatenate(V)
    C=sp.coo_matrix((np.r_[V,V],(np.r_[R,Cc],np.r_[Cc,R])),shape=(NI,NI)).tocsr(); C.sum_duplicates()
    return C
def lookup(C,Hh,it):
    """C csr (NI,NI); Hh (n,30) hist items (NI=pad), it (n,) -> (n,30) value C[h,i] (0 pad)"""
    C=C.tocsr(); key=np.repeat(np.arange(NI,dtype=np.int64),np.diff(C.indptr))*NI+C.indices.astype(np.int64)  # sorted
    q=np.minimum(Hh,NI-1).astype(np.int64)*NI+it[:,None].astype(np.int64); pos=np.searchsorted(key,q.ravel()); pos=np.minimum(pos,len(key)-1)
    hit=(key[pos]==q.ravel()); v=np.where(hit,C.data[pos],0).reshape(q.shape)
    v[Hh>=NI]=0; return v
def feats(nm,k,W,tau=None,alpha=.5):
    T=np.load(O+f'tab_{nm}.npz'); H=T['hist']; hts=T['hts']; ts=T['ts']; it=T['item']; n=len(it); rep=np.repeat(np.arange(len(H)),10)
    C=build(k,W,tau=tau); deg=np.asarray(C.sum(1)).ravel()+1.0
    Hr=H[rep]; v=lookup(C,Hr,it)    # (n,30)
    dg=deg[np.minimum(Hr,NI-1)]**alpha*deg[it][:,None]**(1-alpha)
    cn=v/dg; dt=(ts[rep][:,None]-hts[rep]).clip(min=0)/86400.
    out={'sum':cn.sum(1),'max':cn.max(1),'raw':v.sum(1)}
    for tt in (1,3,10): out[f'dec{tt}']=(cn*np.exp(-dt/tt)).sum(1)
    out['top5']=cn[:,:5].sum(1)   # hist is sorted by recency desc
    return out,T['y']
