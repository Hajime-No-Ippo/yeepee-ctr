"""Taste-metric signals: raw per-dimension impression-SVD interactions + a learned bilinear metric (conditional logit on train window).
Leak-safe: impression SVD from events ts<X; bilinear W trained on labels of events ts<X only (in-sample impression vectors; label-free vectors)."""
import sys; sys.dont_write_bytecode=True
sys.path.insert(0,'/workspace/huawei/dw_v12sig'); sys.path.insert(0,'/workspace/huawei/dw_signals')
from sigs import *
from sigs import _zwithin
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

def imp_svd(D,X,k=24):
    nX=int(np.searchsorted(D.ETS,X)); u=np.repeat(D.EU[:nX],10); it=D.RI[:nX].ravel()
    M=csr_matrix((np.ones(len(u),np.float32),(u,it)),shape=(D.NU,D.NI)); M.sum_duplicates()
    nu=np.asarray(M.sum(1)).ravel(); ni=np.asarray(M.sum(0)).ravel(); Mc=M.tocoo()
    Ms=csr_matrix((Mc.data/np.sqrt((nu[Mc.row]+1)*(ni[Mc.col]+1)),(Mc.row,Mc.col)),shape=M.shape,dtype=np.float32)
    U,sv,Vt=svds(Ms,k=k,random_state=0); o=np.argsort(-sv); U,sv,Vt=U[:,o],sv[o],Vt[o]
    return U,sv,Vt.T,nu

def learn_bilinear(D,X,Un,Vn,nu,nev=300000,epochs=3,seed=0,lam=1e-4):
    """conditional logit with outside option: score_ij=u^T W v + b_i.  W init = I*3. events ts in [X-7d,X) sampled."""
    torch.manual_seed(seed); torch.set_num_threads(3)
    nX=int(np.searchsorted(D.ETS,X)); s=int(np.searchsorted(D.ETS,X-7*DAY)); rs=np.random.RandomState(seed)
    ev=rs.choice(np.arange(s,nX),min(nev,nX-s),replace=False); ev=ev[nu[D.EU[ev]]>0]
    k=Un.shape[1]; W=torch.nn.Parameter(torch.eye(k)*3.0); b0=torch.nn.Parameter(torch.zeros(1)); out0=torch.nn.Parameter(torch.zeros(1))
    Ut=torch.tensor(Un); Vt=torch.tensor(Vn); opt=torch.optim.Adam([W,b0,out0],lr=0.02)
    for ep in range(epochs):
        p=rs.permutation(len(ev))
        for i in range(0,len(p),8192):
            e=ev[p[i:i+8192]]; u=Ut[D.EU[e]]; v=Vt[torch.tensor(D.RI[e].astype(np.int64))]   # (B,k),(B,10,k)
            sc=torch.einsum('bk,kl,bjl->bj',u,W,v); y=torch.tensor(D.Y[e].astype(np.float32)); clicked=y.sum(1)
            logits=torch.cat([sc,(out0+0*b0).expand(len(e),1)],1); tgt=torch.where(clicked>0,y.argmax(1),torch.full_like(clicked,10,dtype=torch.long).long())
            loss=torch.nn.functional.cross_entropy(logits,tgt)+lam*((W-torch.eye(k)*3)**2).sum(); opt.zero_grad(); loss.backward(); opt.step()
    return W.detach().numpy()

def taste_metric(D,X,rows,k=24):
    U,sv,V,nu=imp_svd(D,X,k); Un=norm(U*sv); Vn=norm(V*sv); u=rows['user']; it=rows['item']; out={}
    has=(nu[np.minimum(u,len(nu)-1)]>0)&(u<len(nu))
    uu=np.where(has[:,None],Un[np.minimum(u,len(nu)-1)],0); vv=Vn[it]
    P=uu*vv   # per-dim products (first 12 dims are the main taste dims)
    for d in range(12): out[f'tm_prod{d}']=np.where(has,P[:,d],np.nan).astype(np.float32)
    out['tm_prod_tail']=np.where(has,P[:,12:].sum(1),np.nan).astype(np.float32)
    for d in range(12): out[f'tm_item{d}']=Vn[it,d]
    W=learn_bilinear(D,X,Un,Vn,nu,k=None) if False else learn_bilinear(D,X,Un,Vn,nu)
    sc=np.einsum('rk,kl,rl->r',uu,W,vv); sc=np.where(has,sc,np.nan).astype(np.float32); out['tm_bil']=sc; z,dm=_zwithin(sc); out['tm_bil_z']=z; out['tm_bil_dm']=dm
    out['tm_cos']=np.where(has,(uu*vv).sum(1),np.nan).astype(np.float32); out['tm_bil_minus_cos']=out['tm_bil']-out['tm_cos']*3
    return out

if __name__=='__main__':
    import time; c=int(sys.argv[1]); D=Data(); t=time.time()
    if c==27:
        nt=D.NE; V=extend_eval(D); assert int(np.searchsorted(D.ETS,Jd(27)))==nt
    else: V=D.val(c)
    df=pl.DataFrame({k:np.asarray(v,np.float32) for k,v in taste_metric(D,Jd(c),V).items()}).with_columns(pl.Series('rowid',V['rowid']))
    df.write_parquet(f'/workspace/huawei/dw_v12sig/cache/tm_c{c}.parquet'); print(c,df.shape,round(time.time()-t),'s',flush=True)
