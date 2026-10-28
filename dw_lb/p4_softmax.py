"""Priority 4: parametric single-choice softmax + user-specific outside option, trained end-to-end on ALL events with ts<C (torch, CPU).
score_j = <u,v_j> + b_j + g(match(j in history)); outside logit o = MLP(u,...). P(click j)=exp(s_j)/(exp(o)+sum exp s_k). Outputs val logit log P(click j)."""
import sys, time; sys.path.insert(0,'/workspace/huawei/dw_lb')
from lib import *
import torch, torch.nn as nn, torch.nn.functional as Fn
import dwcommon as D
torch.set_num_threads(4); import os; USEU=int(os.environ.get('USEU','1')); TIE=int(os.environ.get('TIE','0'))
 
def pad_hist(d,ev,L=20):
    n=len(ev); Hp=np.full((n,L),-1,np.int64); Tp=np.zeros((n,L),np.float32)
    for i,e in enumerate(ev):
        a,b=d.OFF[e],d.OFF[e+1]; k=b-a
        if k: Hp[i,L-k:]=d.HI[a:b]; Tp[i,L-k:]=(d.ETS[e]-d.HT[a:b])/3600.
    return Hp,Tp
class M(nn.Module):
    def __init__(s,NU,NI,k=32):
        super().__init__(); s.Eu=nn.Embedding(NU,k); s.Ei=nn.Embedding(NI,k); s.Eh=nn.Embedding(NI+1,k,padding_idx=NI); s.bi=nn.Embedding(NI,1)
        s.pw=nn.Parameter(torch.zeros(20)); s.proj=nn.Linear(k,k); s.match=nn.Sequential(nn.Linear(4,16),nn.ReLU(),nn.Linear(16,1))
        s.out=nn.Sequential(nn.Linear(k+3,32),nn.ReLU(),nn.Linear(32,1)); s.k=k
        for e in (s.Eu,s.Ei,s.Eh): nn.init.normal_(e.weight,0,0.05)
        nn.init.zeros_(s.bi.weight)
    def forward(s,u,S,Hp,Tp,gap):
        if TIE: s.Eh.weight.data[:-1]=s.Ei.weight.data if False else s.Eh.weight.data[:-1]
        NI=s.Ei.num_embeddings; valid=(Hp>=0); Hq=torch.where(valid,Hp,torch.full_like(Hp,NI))
        w=torch.softmax(s.pw[None]+torch.where(valid,0.,-1e4),1)*valid.any(1,keepdim=True)
        pool=(s.Eh(Hq)*w[:,:,None]).sum(1); uv=(s.Eu(u) if USEU else 0)+s.proj(pool)
        sc=(s.Ei(S)*uv[:,None,:]).sum(2)+s.bi(S).squeeze(2)
        m=(S[:,:,None]==Hp[:,None,:])&valid[:,None,:]; cnt=m.sum(2).float(); pos=torch.where(m,torch.arange(20,device=S.device)[None,None,:].float(),torch.full((1,),-1.,device=S.device)).max(2).values; rec=torch.where(cnt>0,(pos+1)/20.,torch.zeros_like(pos))
        age=torch.where(m,Tp[:,None,:],torch.zeros_like(Tp[:,None,:])).sum(2)/torch.clamp(cnt,min=1)
        mf=torch.stack([(cnt>0).float(),torch.log1p(cnt),rec,torch.log1p(age)/5],2); sc=sc+s.match(mf).squeeze(2)
        hl=valid.sum(1,keepdim=True).float()
        o=s.out(torch.cat([uv,hl/20,torch.log1p(gap)[:,None]/5,(hl==0).float()],1)).squeeze(1)
        return sc,o
def run(c,epochs=6,k=32,seed=0,lam=1e-2,lr=3e-3,tag=''):
    torch.manual_seed(seed); d=D.Data(); C=Jd(c); nX=int(np.searchsorted(d.ETS,C)); v=d.val(c); ev=v['e'][::10]
    tr=np.arange(nX); Htr,Ttr=pad_hist(d,tr); Hv,Tv=pad_hist(d,ev)
    gap=lambda e: np.zeros(len(e),np.float32)
    ys=d.Y[:nX]; lab=np.where(ys.sum(1)>0,ys.argmax(1)+1,0)   # 0=no click(outside)
    S=torch.tensor(d.RI[:nX].astype(np.int64)); U=torch.tensor(d.EU[:nX].astype(np.int64)); Hp=torch.tensor(Htr); Tp=torch.tensor(Ttr); L=torch.tensor(lab); G=torch.zeros(nX)
    Sv=torch.tensor(v['item'].reshape(-1,10).astype(np.int64)); Uv=torch.tensor(v['user'][::10].astype(np.int64)); Hvt=torch.tensor(Hv); Tvt=torch.tensor(Tv); Gv=torch.zeros(len(ev))
    m=M(d.NU,d.NI,k); opt=torch.optim.Adam(m.parameters(),lr=lr); yv=v['y']
    def predict():
        m.eval(); out=[]
        with torch.no_grad():
            for i in range(0,len(ev),8192):
                sc,o=m(Uv[i:i+8192],Sv[i:i+8192],Hvt[i:i+8192],Tvt[i:i+8192],Gv[i:i+8192]); lse=torch.logsumexp(torch.cat([o[:,None],sc],1),1,keepdim=True); out.append((sc-lse).numpy())
        m.train(); return np.concatenate(out)
    bs=2048; t0=time.time()
    for epc in range(epochs):
        perm=torch.randperm(nX)
        for i in range(0,nX,bs):
            b=perm[i:i+bs]; sc,o=m(U[b],S[b],Hp[b],Tp[b],G[b]); logit=torch.cat([o[:,None],sc],1); loss=Fn.cross_entropy(logit,L[b])+lam*((m.Eu(U[b])**2).sum(1).mean()+(m.Ei(S[b])**2).sum(2).mean()+(m.Eh.weight[Hp[b].clamp(min=0)]**2).sum(2).mean())
            opt.zero_grad(); loss.backward(); opt.step()
        p=predict(); print(f'c{c} ep{epc} loss {loss.item():.4f} val AUC logP {A(yv,p.ravel()):.5f}  t={time.time()-t0:.0f}s',flush=True)
    np.save(f'/workspace/huawei/dw_lb/cache_sm{tag}_c{c}.npy',p.ravel().astype(np.float32))
if __name__=='__main__': run(int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),0,float(sys.argv[4]),float(sys.argv[5]),sys.argv[6] if len(sys.argv)>6 else '')
