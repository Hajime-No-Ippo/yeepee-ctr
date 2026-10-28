"""v16seq: next-click sequence models (SASRec/gBCE, GRU4Rec) as feature generators. All sequences strictly before the fold cutoff C (click log of train events ts<C, i.e. their histories + their clicks)
plus the TARGET event's own history array (known at event time). No eval-row aggregates, no labels of events >= C.
Usage (python seqfeat.py MODEL CUTOFF_NAME TABLES... ): see seqfeat.py."""
import os, sys, time, math, numpy as np, polars as pl, torch, torch.nn as nn, torch.nn.functional as F
os.environ.setdefault('DW_WORK','/workspace/huawei/v12x/dwdata'); sys.path.insert(0,'/workspace/huawei/v16seq'); sys.dont_write_bytecode=True
from common import *
L=50; F3='/workspace/huawei/v12/feat_v3'; DW='/workspace/huawei/v12x/dwdata'
CUTS={**{f'C{k}':J20+k*DAY for k in range(1,8)}}
TABLE_CUT={'f1':'C1','f2':'C2','f3':'C3','f4':'C4','f5':'C5','f6':'C6','val23':'C3','val25':'C5','val26':'C6','eval':'C7'}
_D=None
def data():
    global _D
    if _D is None: _D=Data()
    return _D
def gapb(sec):  # log2-minute bucket 1..21 (0 = pad)
    return 1+np.clip(np.log2(np.maximum(sec,60)/60.),0,19).astype(np.int64)
def table_events(name):
    """-> dict(rowid, u[n_ev] (user idx or -1), ts, hist list of (items idx array, ts array), cand[n_ev,10] item idx (NI+1 = unknown))"""
    D=data(); v=pl.read_parquet(f'{F3}/{name}.parquet',columns=['rowid']); rid=v['rowid'].to_numpy()
    if name=='eval':
        ee=pl.read_parquet(f'{DW}/evaluation_events.parquet'); er=pl.read_parquet(f'{DW}/evaluation_rows.parquet')
        m=v.join(er.select('rowid','eid','item'),on='rowid',how='left',maintain_order='left'); assert m['eid'].null_count()==0
        eid=m['eid'].to_numpy().reshape(-1,10); assert (eid==eid[:,:1]).all(); eids=eid[:,0]
        e2=pl.DataFrame({'eid':eids}).join(ee,on='eid',how='left',maintain_order='left'); assert e2['ts'].null_count()==0
        ui=np.searchsorted(D.uu,e2['user'].to_numpy()); ok=(ui<D.NU)&(D.uu[np.minimum(ui,D.NU-1)]==e2['user'].to_numpy()); u=np.where(ok,ui,-1)
        ts=e2['ts'].to_numpy(); hi=e2['h_items'].to_list(); ht=e2['h_ts'].to_list()
        def mp(a): a=np.asarray(a,dtype=D.items.dtype) if len(a) else np.zeros(0,D.items.dtype); p=np.minimum(np.searchsorted(D.items,a),D.NI-1); return np.where(D.items[p]==a,p,D.NI+1) if len(a) else np.zeros(0,np.int64)
        H=[(mp(a or []),np.asarray(b or [],np.int64)) for a,b in zip(hi,ht)]
        ci=m['item'].to_numpy(); p=np.minimum(np.searchsorted(D.items,ci),D.NI-1); cand=np.where(D.items[p]==ci,p,D.NI+1).reshape(-1,10)
    else:
        m=v.join(D.rows_rowid,on='rowid',how='left',maintain_order='left'); assert m['e'].null_count()==0
        e=m['e'].to_numpy().astype(np.int64).reshape(-1,10); assert (e==e[:,:1]).all(); es=e[:,0]
        u=D.EU[es].astype(np.int64); ts=D.ETS[es]; H=[(D.HI[D.OFF[x]:D.OFF[x+1]].astype(np.int64),D.HT[D.OFF[x]:D.OFF[x+1]]) for x in es]
        cand=np.searchsorted(D.items,m['item'].to_numpy()).astype(np.int64).reshape(-1,10)
    return dict(rowid=rid,u=u,ts=ts,H=H,cand=cand)
def click_seqs(C):
    """per-user click sequences (items, minute-times) from train events with ts<C. -> (offsets by user, items, times)"""
    D=data(); S=D.snap(C); cu,ci,ct=S['cu'].astype(np.int64),S['ci'].astype(np.int64),S['ct'].astype(np.int64)
    o=np.lexsort((ci,ct,cu)); cu,ci,ct=cu[o],ci[o],ct[o]; off=np.zeros(D.NU+1,np.int64); off[1:]=np.cumsum(np.bincount(cu,minlength=D.NU)); return off,ci,ct
def windows(off,it,tm,stride=25):
    """training windows of L+1 consecutive clicks; right-anchored padding. returns inp,tgt,gap [N,L] int64 (pad item = NI)"""
    D=data(); NI=D.NI; inp=[];tgt=[];gp=[]
    for u in range(len(off)-1):
        a,b=off[u],off[u+1]; n=b-a
        if n<2: continue
        ends=list(range(n,0,-stride)); 
        for e in ends:
            s=max(0,e-(L+1)); w=it[a+s:a+e]; t=tm[a+s:a+e]; k=len(w)-1
            if k<1: continue
            i=np.full(L,NI,np.int64); g=np.zeros(L,np.int64); y=np.full(L,NI,np.int64)
            i[L-k:]=w[:-1]; y[L-k:]=w[1:]; g[L-k:]=gapb(t[1:]-t[:-1]); inp.append(i);tgt.append(y);gp.append(g)
    return np.stack(inp),np.stack(tgt),np.stack(gp)
class Block(nn.Module):
    def __init__(s,d,H,dr):
        super().__init__(); s.H=H; s.n1=nn.LayerNorm(d); s.qkv=nn.Linear(d,3*d); s.o=nn.Linear(d,d); s.n2=nn.LayerNorm(d); s.f=nn.Sequential(nn.Linear(d,2*d),nn.GELU(),nn.Dropout(dr),nn.Linear(2*d,d)); s.dr=nn.Dropout(dr)
    def forward(s,x,mask):
        B,T,d=x.shape; q,k,v=s.qkv(s.n1(x)).view(B,T,3,s.H,d//s.H).permute(2,0,3,1,4)
        a=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=s.dr.p if s.training else 0.).transpose(1,2).reshape(B,T,d)
        x=x+s.dr(s.o(a)); return x+s.dr(s.f(s.n2(x)))
class Seq(nn.Module):
    def __init__(s,NI,d=64,kind='sas',nb=2,H=2,dr=0.2):
        super().__init__(); s.kind=kind; s.NI=NI; s.E=nn.Embedding(NI+2,d,padding_idx=NI); nn.init.normal_(s.E.weight,std=d**-0.5); s.G=nn.Embedding(22,d,padding_idx=0); s.dr=nn.Dropout(dr)
        if kind=='sas': s.P=nn.Embedding(L,d); s.B=nn.ModuleList([Block(d,H,dr) for _ in range(nb)]); s.nf=nn.LayerNorm(d)
        else: s.gru=nn.GRU(d,d,batch_first=True); s.nf=nn.LayerNorm(d)
    def forward(s,it,gp):
        x=s.E(it)+s.G(gp)
        if s.kind=='sas':
            B,T=it.shape; x=s.dr(x+s.P.weight[None,:T]); pad=(it==s.NI); i=torch.arange(T)
            allow=(i[None,:]<=i[:,None])[None]&(~pad)[:,None,:]; allow=allow|torch.eye(T,dtype=torch.bool)[None]
            m=allow[:,None]; 
            for b in s.B: x=b(x,m)
        else:
            x=s.dr(x); x,_=s.gru(x)
        return s.nf(x)
def train(kind,C,epochs=12,K=256,t=0.75,bs=256,lr=2e-3,seed=0,d=64,loss='gbce',log_=print,maxsec=None):
    D=data(); NI=D.NI; torch.manual_seed(seed); np.random.seed(seed); off,it,tm=click_seqs(C); inp,tgt,gp=windows(off,it,tm); N=len(inp)
    inp,tgt,gp=map(torch.from_numpy,(inp,tgt,gp)); m=Seq(NI,d,kind); opt=torch.optim.Adam(m.parameters(),lr=lr); t0=time.time()
    alpha=K/(NI-1); beta=alpha*(t*(1-1/alpha)+1/alpha); nstep=epochs*math.ceil(N/bs); sch=torch.optim.lr_scheduler.OneCycleLR(opt,lr,total_steps=nstep,pct_start=.1)
    for ep in range(epochs):
        m.train(); perm=torch.randperm(N); tl=0;nb=0
        for b in range(0,N,bs):
            j=perm[b:b+bs]; x,y,g=inp[j],tgt[j],gp[j]; h=m(x,g); msk=y!=NI; hv=h[msk]; yv=y[msk]
            neg=torch.randint(0,NI,(K,)); En=m.E.weight[neg]; pos=(hv*m.E.weight[yv]).sum(-1,keepdim=True); ng=hv@En.T; hit=(yv[:,None]==neg[None])
            if loss=='gbce':
                ng=ng.masked_fill(hit,-1e4); p=torch.sigmoid(pos.double()).clamp(1e-10,1-1e-10); pa=p.pow(-beta).clamp(min=1+1e-10); pl_=torch.log((1/(pa-1)).clamp(min=1e-10)).float()
                lg=torch.cat([pl_,ng],1); lab=torch.zeros_like(lg); lab[:,0]=1; l=F.binary_cross_entropy_with_logits(lg,lab,reduction='none'); l=l.sum(1).mean()
            else:
                ng=ng.masked_fill(hit,-1e4); l=F.cross_entropy(torch.cat([pos,ng],1),torch.zeros(len(hv),dtype=torch.long))
            opt.zero_grad(); l.backward(); nn.utils.clip_grad_norm_(m.parameters(),5.); opt.step(); sch.step(); tl+=l.item(); nb+=1
        log_(f'  [{kind} C={(C-J20)//DAY}] ep{ep} loss {tl/nb:.4f} N={N} {time.time()-t0:.0f}s')
    m.eval(); return m
def states(m,tab,C,sub):
    """user state at last position for full history, last-1 and last-3 tokens. returns [n_ev,3,d] (full,last1,last3) """
    D=data(); NI=D.NI; off,it,tm=click_seqs(C); n=len(tab['u']); ITs=np.full((3,n,L),NI,np.int64); Gs=np.zeros((3,n,L),np.int64)
    for i in range(n):
        u=tab['u'][i]; hi,ht=tab['H'][i]; ts=tab['ts'][i]
        if u>=0: a,b=off[u],off[u+1]; li,lt=it[a:b],tm[a:b]
        else: li=np.zeros(0,np.int64); lt=np.zeros(0,np.int64)
        ok=ht<ts; hi,ht=hi[ok],ht[ok]   # history clicks strictly before the event (defensive)
        si=np.concatenate([li,hi]); st=np.concatenate([lt,ht])
        if len(si):
            key=si.astype(np.int64)*(1<<22)+(st-J20+DAY*30)//60; _,ix=np.unique(key,return_index=True); ix=np.sort(ix); si,st=si[ix],st[ix]; o=np.argsort(st,kind='stable'); si,st=si[o],st[o]
            si,st=si[-L:],st[-L:]; k=len(si); g=gapb(np.concatenate([st[1:],[ts]])-st)
            for r,kk in enumerate((L,1,3)):
                kk=min(kk,k); ITs[r,i,L-kk:]=si[k-kk:]; Gs[r,i,L-kk:]=g[k-kk:]
    out=np.zeros((n,3,m.E.weight.shape[1]),np.float32)
    with torch.no_grad():
        for r in range(3):
            for b in range(0,n,4096): out[b:b+4096,r]=m(torch.from_numpy(ITs[r,b:b+4096]),torch.from_numpy(Gs[r,b:b+4096]))[:,-1].numpy()
    has=np.array([len(h[0])>0 or (u>=0 and off[u+1]>off[u]) for u,h in zip(tab['u'],tab['H'])]); return out,has
def featurize(m,tab,C,name):
    st,has=states(m,tab,C,name); W=m.E.weight.detach().numpy(); cand=tab['cand']; Ec=W[cand]  # [n,10,d]
    def sc(h): return np.einsum('nd,ncd->nc',h,Ec)
    def cs(h): return sc(h)/((np.linalg.norm(h,axis=1,keepdims=True)+1e-6)*(np.linalg.norm(Ec,axis=2)+1e-6))
    S=sc(st[:,0]); cols={}
    mu=S.mean(1,keepdims=True); sd=S.std(1,keepdims=True)+1e-6; ls=S-np.log(np.exp(S-S.max(1,keepdims=True)).sum(1,keepdims=True))-S.max(1,keepdims=True)
    cols['dot']=S; cols['cos']=cs(st[:,0]); cols['z']=(S-mu)/sd; cols['gap']=S-S.max(1,keepdims=True); cols['lsm']=ls
    cols['rank']=(np.argsort(np.argsort(-S,1,kind='stable'),1)+1).astype(np.float32)
    for r,nm in ((1,'1'),(2,'3')):
        s_=sc(st[:,r]); cols['dot'+nm+'_dm']=s_-s_.mean(1,keepdims=True); cols['cos'+nm]=cs(st[:,r])
    out={k:np.where(has[:,None],v,np.nan).astype(np.float32).ravel() for k,v in cols.items()}; return out
