"""Two-tower click model trained on events before cutoff: user = position-weighted sum of
history item embeddings; score = <user, item_out> + item bias. Trained with BCE per candidate
plus an in-event softmax term (with a learned 'no click' option)."""
import torch, numpy as np, polars as pl, time
torch.set_num_threads(4)
H=20
def pack(E):
    """events -> padded history matrix [n,H] (newest last, right-aligned), mask, plus delta-time buckets"""
    n=len(E); M=np.full((n,H),-1,np.int64); DT=np.zeros((n,H),np.float32)
    hl=E['h'].to_list(); htl=E['h_ts'].to_list(); ts=E['ts'].to_numpy()
    for i,(h,t) in enumerate(zip(hl,htl)):
        if h:
            h=h[-H:]; t=t[-H:]; M[i,H-len(h):]=h; DT[i,H-len(h):]=np.log1p((ts[i]-np.array(t))/3600.)
    return M,DT
class TT(torch.nn.Module):
    def __init__(s,NI,d=64):
        super().__init__()
        s.ein=torch.nn.Embedding(NI+1,d,padding_idx=NI); s.eout=torch.nn.Embedding(NI,d); s.b=torch.nn.Embedding(NI,1)
        s.pw=torch.nn.Parameter(torch.zeros(H)); s.tw=torch.nn.Parameter(torch.zeros(1)); s.b0=torch.nn.Parameter(torch.tensor(-2.7))
        s.null=torch.nn.Linear(d,1)
        for e in (s.ein,s.eout): torch.nn.init.normal_(e.weight,0,0.05)
        torch.nn.init.zeros_(s.b.weight); s.NI=NI
    def user(s,M,DT):
        mask=(M>=0); Mi=torch.where(mask,M,torch.full_like(M,s.NI))
        w=torch.exp(s.pw[None,:]-torch.nn.functional.softplus(s.tw)*DT)*mask
        u=(s.ein(Mi)*w[...,None]).sum(1)/(w.sum(1,keepdim=True)+1e-6)
        return u
    def forward(s,M,DT,C):
        u=s.user(M,DT); return (u[:,None,:]*s.eout(C)).sum(-1)+s.b(C)[...,0]+s.b0, s.null(u)[:,0]
def make_xy(E,R,with_y=True):
    R=R.join(E.select('eid').with_row_index('ev'),on='eid').sort('ev')
    g=R.group_by('ev',maintain_order=True).agg(pl.col('ii'),*( [pl.col('clicked')] if with_y else []))
    C=np.array(g['ii'].to_list(),np.int64); Y=np.array(g['clicked'].to_list(),np.float32) if with_y else None
    return C,Y,R
def train(P_ev,P_rows,NI,epochs=3,d=64,bs=512,lr=3e-3,wd=1e-6,seed=0,log=print):
    torch.manual_seed(seed); np.random.seed(seed)
    M,DT=pack(P_ev); C,Y,_=make_xy(P_ev,P_rows)
    M,DT,C,Y=map(torch.from_numpy,(M,DT,C,Y))
    m=TT(NI,d); opt=torch.optim.Adam(m.parameters(),lr=lr)
    n=len(M); t=time.time()
    for ep in range(epochs):
        perm=torch.randperm(n); tot=0
        for k in range(0,n,bs):
            idx=perm[k:k+bs]; s,nl=m(M[idx],DT[idx],C[idx]); y=Y[idx]
            bce=torch.nn.functional.binary_cross_entropy_with_logits(s,y)
            logits=torch.cat([s,nl[:,None]],1); tgt=torch.where(y.sum(1)>0,y.argmax(1),torch.full((len(y),),10))
            ce=torch.nn.functional.cross_entropy(logits,tgt)
            reg=wd*(m.ein.weight[M[idx].clamp(min=0)].pow(2).sum()+m.eout.weight[C[idx]].pow(2).sum())/len(idx)
            loss=bce+0.1*ce+reg; opt.zero_grad(); loss.backward(); opt.step(); tot+=loss.item()*len(idx)
        log(f'tt epoch {ep} loss {tot/n:.4f} {time.time()-t:.0f}s')
    return m
@torch.no_grad()
def score(m,T_ev,T_rows):
    """returns T_rows (eid,ii) joined with tt_score, tt_null, tt_unorm"""
    m.eval(); M,DT=pack(T_ev); C,_,R=make_xy(T_ev,T_rows,with_y=False)
    s,nl=m(torch.from_numpy(M),torch.from_numpy(DT),torch.from_numpy(C)); u=m.user(torch.from_numpy(M),torch.from_numpy(DT))
    out=pl.DataFrame({'eid':np.repeat(T_ev['eid'].to_numpy(),10),'ii':C.ravel().astype(np.int32),'tt_score':s.numpy().ravel(),
                      'tt_null':np.repeat(nl.numpy(),10),'tt_unorm':np.repeat(u.norm(dim=1).numpy(),10)})
    return out
