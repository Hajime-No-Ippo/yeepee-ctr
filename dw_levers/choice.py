"""Choice model with outside option (CPU PyTorch).  P(click i | slate) = exp(s_i)/(exp(o)+sum_j exp(s_j)), s_i=<u,E_i>+b_i,
u = sum_h w(age_h) * Eh[h] / n^a  (+ optional free user vec).  Trained on train events with ts<X only (+ their own pre-ts histories).
Entry points: fit_choice(D,X,cfg) -> model ; score(model,D,ev_idx) -> (n,10) log-prob (and (n,) log p_none)"""
import numpy as np, torch, torch.nn as nn, time, sys
sys.path.insert(0,'/workspace/huawei/dw_levers')
from dwcommon import *
torch.set_num_threads(8)
HL=20
def build_hist(D):
    if hasattr(D,'H'): return
    n=D.NE; H=np.full((n,HL),D.NI,np.int32); G=np.zeros((n,HL),np.int8)
    rows=np.repeat(np.arange(n),D.Hlen); cols=np.arange(len(D.HI))-np.repeat(D.OFF[:-1],D.Hlen)
    H[rows,cols]=D.HI; gap=np.maximum(D.ETS[rows]-D.HT,60)
    G[rows,cols]=1+np.clip(np.log2(gap/60),0,19).astype(np.int8)
    D.H=H; D.G=G
class CM(nn.Module):
    def __init__(s,NI,NU,V0,d,free_user,att):
        super().__init__(); s.NI=NI
        s.E=nn.Embedding(NI+1,d,padding_idx=NI); s.Eh=nn.Embedding(NI+1,d,padding_idx=NI); s.b=nn.Embedding(NI+1,1,padding_idx=NI)
        with torch.no_grad():
            k=V0.shape[1]; E0=torch.randn(NI+1,d)*0.01; E0[:NI,:k]=torch.from_numpy(V0); E0[NI]=0
            s.E.weight.copy_(E0); s.Eh.weight.copy_(E0); s.b.weight.zero_()
        s.gw=nn.Embedding(HL+2,1); nn.init.zeros_(s.gw.weight)   # age weight logits
        s.scale=nn.Parameter(torch.tensor(3.0)); s.alpha=nn.Parameter(torch.tensor(0.5))
        s.free=nn.Embedding(NU,d) if free_user else None
        if free_user: nn.init.zeros_(s.free.weight)
        s.att=att
        if att: s.Wq=nn.Linear(d,d,bias=False); nn.init.eye_(s.Wq.weight)
        s.out=nn.Sequential(nn.Linear(4,16),nn.ReLU(),nn.Linear(16,1))   # outside logit from user/ctx stats
        s.ob=nn.Embedding(NU,1) if free_user else None
        if free_user: nn.init.zeros_(s.ob.weight)
    def forward(s,H,G,cand,u,ctx):
        valid=(H!=s.NI).float(); nh=valid.sum(1,keepdim=True)
        eh=s.Eh(H); w=torch.sigmoid(s.gw(G.long()).squeeze(-1)+1.0)*valid           # (B,T)
        q=s.E(cand)                                                                # (B,10,d)
        if s.att:
            a=torch.einsum('bcd,btd->bct',s.Wq(q),eh)/eh.shape[-1]**0.5*s.scale; a=a.masked_fill(valid[:,None,:]==0,-1e9)
            wt=torch.softmax(a,-1)*(nh>0).float()[:,:,None]; ur=torch.einsum('bct,btd->bcd',wt*w[:,None,:],eh)  # per-candidate user rep
            sc=(ur*q).sum(-1)*s.scale/ (w.sum(1,keepdim=True)+1e-3).clamp(min=1)**0.0
        else:
            ur=(eh*w[:,:,None]).sum(1)/(nh+1.0)**s.alpha                           # (B,d)
            if s.free is not None: ur=ur+s.free(u)
            sc=torch.einsum('bd,bcd->bc',ur,q)*s.scale
        sc=sc+s.b(cand).squeeze(-1)
        f=torch.cat([torch.log1p(nh),ctx],1)                                       # ctx: (B,3)
        o=s.out(f)+(s.ob(u) if s.ob is not None else 0)
        return sc,o.squeeze(-1)
def ctx_feats(D,ev):
    tb=((D.ETS[ev]-J20)%DAY)/DAY; dt=(((D.ETS[ev]-J20)//DAY))%7/7.0
    nlast=np.where(D.Hlen[ev]>0,D.G[ev].max(1),0)/20.0   # gap-bucket of the oldest? (crude recency) -> use min nonzero below
    G=D.G[ev].astype(np.float32); G[G==0]=99; rec=np.where(D.Hlen[ev]>0,G.min(1),0)/20.0
    return np.stack([tb,dt,rec],1).astype(np.float32)
def fit_choice(D,X,cfg,log_every=1,val=None):
    build_hist(D); t0=time.time(); torch.manual_seed(cfg.get('seed',0)); np.random.seed(cfg.get('seed',0))
    nX=int(np.searchsorted(D.ETS,X)); idx=np.arange(nX)
    d=cfg.get('d',32); S=D.snap(X); V0,_=click_svd(S,D,k=min(d,cfg.get('k',32))); log('svd done')
    V0=V0*cfg.get('v0scale',1.0)
    m=CM(D.NI,D.NU,V0,d,cfg.get('free_user',False),cfg.get('att',False))
    tH=torch.from_numpy(D.H.astype(np.int64)); tG=torch.from_numpy(D.G.astype(np.int64)); tC=torch.from_numpy(D.RI.astype(np.int64)); tU=torch.from_numpy(D.EU.astype(np.int64))
    ctx=torch.from_numpy(ctx_feats(D,np.arange(D.NE)))
    yidx=np.where(D.Y.sum(1)>0,D.Y.argmax(1),10); tY=torch.from_numpy(yidx.astype(np.int64))
    opt=torch.optim.Adam([{'params':[p for n_,p in m.named_parameters() if n_.split('.')[0] not in ('free','ob','E','Eh','b')]},
        {'params':list(m.E.parameters())+list(m.Eh.parameters())+list(m.b.parameters()),'weight_decay':0.0},
        {'params':(list(m.free.parameters())+list(m.ob.parameters())) if m.free is not None else [],'weight_decay':0.0}],lr=cfg.get('lr',2e-3))
    bs=cfg.get('bs',512); wd=cfg.get('l2',0.0); fl2=cfg.get('free_l2',1e-4); ep=cfg.get('ep',4); hid=cfg.get('hist_drop',0.0)
    # held-out tail of fit window for early stopping (last 5% of events by time)
    ntr=int(nX*0.95); hold=np.arange(ntr,nX); idx=np.arange(ntr)
    best=(1e9,None)
    for e in range(ep):
        m.train(); perm=torch.from_numpy(np.random.permutation(idx)); tl=0; nb=0
        for b in range(0,len(perm),bs):
            j=perm[b:b+bs]; H=tH[j]
            if hid>0: H=torch.where(torch.rand(H.shape)<hid,torch.full_like(H,D.NI),H)
            sc,o=m(H,tG[j],tC[j],tU[j],ctx[j]); lg=torch.cat([sc,o[:,None]],1); loss=nn.functional.cross_entropy(lg,tY[j])
            if wd>0:
                loss=loss+wd*((m.E(tC[j])**2).sum(-1).mean()+(m.Eh(H)**2).sum(-1).mean())
            if m.free is not None and fl2>0: loss=loss+fl2*(m.free(tU[j])**2).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tl+=loss.item(); nb+=1
        m.eval(); hl=eval_loss(m,D,hold,tH,tG,tC,tU,ctx,tY); msg=f'  ep{e} train {tl/nb:.4f} hold {hl:.4f} {time.time()-t0:.0f}s'
        if val is not None:
            lp,_=score(m,D,val['e'][::10],tH,tG,tC,tU,ctx); msg+=f" valAUC {A(val['y'],lp.ravel()):.5f}"
        log(msg)
        if hl<best[0]: best=(hl,{k:v.clone() for k,v in m.state_dict().items()})
    m.load_state_dict(best[1]); m.eval(); m.cache=(tH,tG,tC,tU,ctx); return m
@torch.no_grad()
def eval_loss(m,D,ev,tH,tG,tC,tU,ctx,tY):
    tot=0
    for b in range(0,len(ev),8192):
        j=torch.from_numpy(ev[b:b+8192]); sc,o=m(tH[j],tG[j],tC[j],tU[j],ctx[j]); tot+=nn.functional.cross_entropy(torch.cat([sc,o[:,None]],1),tY[j],reduction='sum').item()
    return tot/len(ev)
@torch.no_grad()
def score(m,D,ev,tH=None,tG=None,tC=None,tU=None,ctx=None):
    if tH is None: tH,tG,tC,tU,ctx=m.cache
    out=[];on=[]
    for b in range(0,len(ev),8192):
        j=torch.from_numpy(np.asarray(ev[b:b+8192])); sc,o=m(tH[j],tG[j],tC[j],tU[j],ctx[j]); lg=torch.cat([sc,o[:,None]],1); lp=torch.log_softmax(lg,1); out.append(lp[:,:10].numpy()); on.append(lp[:,10].numpy())
    return np.concatenate(out),np.concatenate(on)
