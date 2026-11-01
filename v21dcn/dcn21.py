"""seq21.py CUT TAG --var A|B [--enc din|gru|trf] [--seeds 0,1,2] [--ep 4] ...
v21seq: end-to-end listwise model = tabular MLP (same matrix as mlp21list/v21list) + learned item-ID embeddings (init from train-only PPMI-SVD at the earliest
training cutoff) + DIN-style target attention over the user's own pre-ts click history (<=20, newest first, time-gap/position enc, frozen PPMI-SVD cosines at the
table's cutoff as pair features, optional GRU/transformer history encoder) + slate self-attention over the 10 candidates, listwise slate softmax with outside option.
Saves ens/{TAG}_c{CUT}_s{seed}.npy (n, EP) log-probs after each epoch (row order of the cached table)."""
import os, sys, re, time, argparse, json, gc, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0,'/workspace/huawei/v19'); sys.path.insert(0,'/workspace/huawei/v21seq'); os.chdir('/workspace/huawei/v21dcn'); sys.dont_write_bytecode=True
from cl import matrix
from sklearn.metrics import roc_auc_score
ap=argparse.ArgumentParser(); ap.add_argument('cut',type=int); ap.add_argument('tag'); ap.add_argument('--var',default='A'); ap.add_argument('--enc',default='din')
ap.add_argument('--seeds',default='0'); ap.add_argument('--ep',type=int,default=4); ap.add_argument('--hid',type=int,default=512); ap.add_argument('--drop',type=float,default=0.2)
ap.add_argument('--d',type=int,default=32); ap.add_argument('--hd',type=int,default=64); ap.add_argument('--tok',type=int,default=192); ap.add_argument('--slayers',type=int,default=2)
ap.add_argument('--edrop',type=float,default=0.1); ap.add_argument('--aug',type=float,default=0.0); ap.add_argument('--alpha',type=float,default=0.0)
ap.add_argument('--lr',type=float,default=2e-3); ap.add_argument('--elr',type=float,default=1e-3); ap.add_argument('--bs',type=int,default=256); ap.add_argument('--noseq',action='store_true'); ap.add_argument('--noslate',action='store_true')
ap.add_argument('--nf',type=int,default=0); ap.add_argument('--arch',default='mlp'); ap.add_argument('--cl',type=int,default=3); ap.add_argument('--maxtrain',type=int,default=0); ap.add_argument('--teacher',default=''); ap.add_argument('--tw',type=float,default=1.0); ap.add_argument('--outdir',default='/workspace/huawei/v21dcn/ens')
ap.add_argument('--needseq',type=int,default=0); ap.add_argument('--cfg',default='',help='json file: list of {name,seeds,<overrides>} run in one process (data loaded once)')
ap.add_argument('--cr',type=int,default=0,help='low-rank cross rank (0=full)'); ap.add_argument('--stack',type=int,default=0); ap.add_argument('--fmk',type=int,default=0)
ap.add_argument('--wd',type=float,default=1e-4); ap.add_argument('--lw',type=float,default=1.0); ap.add_argument('--cos',type=int,default=0); ap.add_argument('--candid',type=int,default=0)
ap.add_argument('--pct',type=float,default=0.1); ap.add_argument('--ls',type=float,default=0.0); ap.add_argument('--gate',type=int,default=0); ap.add_argument('--xdrop',type=float,default=0.0); ap.add_argument('--noise',type=float,default=0.0)
a=ap.parse_args(); t0=time.time(); torch.set_num_threads(int(os.environ.get('MLP_THREADS',4)))
DR='^sq_(pdot|pcos|mdot|mcos)$'
lrx=f'feat_lr2,feat_tt,feat_pp,feat_sqC,feat_dw,feat_{a.var}p,feat_N{a.var}s'; extra=tuple(lrx.split(','))
trn=[f'f{k}' for k in range(max(1,a.cut-24),a.cut-20)]; vn='eval' if a.cut==27 else f'val{a.cut}'
if a.nf>0: trn=trn[-a.nf:]
def seqload(n):
    z=np.load(f'/workspace/huawei/v21seq/seqfeat/{a.var}_{n}.npz',allow_pickle=True); return {k:z[k] for k in z.files}
def realign(s,rid):
    if (s['rowid']==rid).all(): return s
    sr=np.asarray(s['rowid']).astype(str); o_=np.argsort(sr); assert len(np.unique(sr))==len(sr); pos=np.searchsorted(sr[o_],np.asarray(rid).astype(str)); p=o_[pos]; assert (sr[p]==np.asarray(rid).astype(str)).all()
    ev=p.reshape(-1,10)//10; assert (ev==ev[:,:1]).all(); ev=ev[:,0]
    o=dict(s); 
    for k in ('hidx','hgap','hlen','ets'): o[k]=s[k][ev]
    for k in ('cand','cos1','cos2','rowid'): o[k]=s[k][p]
    print('realigned',len(rid),flush=True); return o
Xs=[];ys=[];SQ=[]
for n in trn:
    X,m,f=matrix(n,1.0,extra=extra); y=m['clicked'].to_numpy(); feats=f; keep=[i for i,c in enumerate(feats) if not re.search(DR,c)]
    if a.needseq: s=seqload(n); assert (s['rowid']==m['rowid'].to_numpy()).all(),n
    else: zz=np.load(f'/workspace/huawei/v21seq/seqfeat/{a.var}_{n}.npz',allow_pickle=True); assert (zz['rowid']==m['rowid'].to_numpy()).all(),n; s=dict(NI=int(zz['NI']),hidx=np.zeros((len(y),20),np.int32),hgap=np.zeros((len(y),20),np.float32),cos1=np.zeros((len(y)*10,1),np.float16),cos2=np.zeros((len(y)*10,1),np.float16),cand=zz['cand'])
    SQ.append(s)
    Xs.append(np.ascontiguousarray(X[:,keep])); ys.append(y); del X; gc.collect()
fl=[feats[i] for i in keep]; NR=sum(len(a_) for a_ in Xs); X=np.empty((NR,Xs[0].shape[1]),np.float32); o_=0
while Xs: a_=Xs.pop(0); X[o_:o_+len(a_)]=a_; o_+=len(a_); del a_; gc.collect()
y=np.concatenate(ys).astype(np.float32)
Xv,mv,fv=matrix(vn,1.0,extra=extra); Xv=Xv[:,[fv.index(c) for c in fl]]; sv=seqload(vn); sv=realign(sv,mv['rowid'].to_numpy())
NI=int(SQ[0]['NI']); 
cat=lambda k: np.concatenate([s[k] for s in SQ])
HI=torch.from_numpy(cat('hidx').astype(np.int64)); HG=torch.from_numpy(cat('hgap')); C1=torch.from_numpy(cat('cos1')); C2=torch.from_numpy(cat('cos2')); CAND=torch.from_numpy(cat('cand').astype(np.int64))
HIv=torch.from_numpy(sv['hidx'].astype(np.int64)); HGv=torch.from_numpy(sv['hgap']); C1v=torch.from_numpy(sv['cos1']); C2v=torch.from_numpy(sv['cos2']); CANDv=torch.from_numpy(sv['cand'].astype(np.int64))
init=np.load(f'/workspace/huawei/v21seq/seqfeat/init_A_f{trn[0][1:]}.npy' if os.path.exists(f'/workspace/huawei/v21seq/seqfeat/init_A_f{trn[0][1:]}.npy') else '/workspace/huawei/v21seq/seqfeat/init_A_f1.npy').astype(np.float32)   # TRAIN-ONLY graph at the earliest training cutoff (no label of any training row in it)
init=init/(np.linalg.norm(init,axis=1,keepdims=True)+1e-9)
from scipy.special import ndtri
qs=np.linspace(0.001,0.999,101); Q=np.nanquantile(X[::7],qs,axis=0).astype(np.float32)
nanc=np.where(np.isnan(X[::7]).any(0)|np.isnan(Xv[::7]).any(0))[0]; G=ndtri(qs).astype(np.float32)
def tr(Z):
    fl_=np.isnan(Z[:,nanc])
    for j in range(Z.shape[1]):
        q=np.nan_to_num(Q[:,j],nan=0.0); v=Z[:,j]; u,ix=np.unique(q,return_index=True); nn_=np.isnan(v)
        o=np.interp(np.nan_to_num(v),u,G[ix]) if len(u)>1 else np.zeros_like(v); o[nn_]=0; Z[:,j]=o
    return Z,fl_
X,FX=tr(X); gc.collect(); Xv,FV=tr(Xv); gc.collect(); print('prep',X.shape,FX.shape,round(time.time()-t0),'s',flush=True)
D_=X.shape[1]+len(nanc); E=len(y)//10; Ev=len(Xv)//10
Xt=torch.from_numpy(X); Ft=torch.from_numpy(FX); Yt=torch.from_numpy(y); Xvt=torch.from_numpy(Xv); Fvt=torch.from_numpy(FV)

TW=None
if a.teacher:
    T=np.load(a.teacher).astype(np.float32)   # (Ev*10,) teacher PROBABILITIES per row, matrix row order of the val/eval table
    assert len(T)==len(Xv)
    Xt=torch.cat([Xt,Xvt]); Ft=torch.cat([Ft,Fvt]); HI=torch.cat([HI,HIv]); HG=torch.cat([HG,HGv]); C1=torch.cat([C1,C1v]); C2=torch.cat([C2,C2v]); CAND=torch.cat([CAND,CANDv])
    Yt=torch.cat([Yt,torch.from_numpy(T)]); E0_=E; E=E+Ev
    ISP=torch.cat([torch.zeros(E0_),torch.ones(Ev)])
    print('distill: real events',E0_,'pseudo events',Ev,'mean teacher p',float(T.mean()),flush=True)
else: ISP=torch.zeros(E)
def tf_feats(hg,mask):   # hg (B,20) seconds newest-first
    g=hg.clamp(min=0); lg=torch.log1p(g/3600.)/6; ld=torch.log1p(g/86400.)/3
    prev=torch.cat([torch.zeros_like(g[:,:1]),g[:,:-1]],1); dif=torch.log1p((g-prev).clamp(min=0)/3600.)/6   # time between this click and the next newer one
    pos=torch.arange(20,device=hg.device).float()[None].expand_as(g)/20
    return torch.stack([lg,ld,dif,pos],-1)*mask[...,None]
class Net(nn.Module):
    def __init__(s,D,h,dr):
        super().__init__(); d=a.d; hd=a.hd; tk=a.tok
        if a.arch=='dcn':
            if a.cr>0: s.cw=nn.ModuleList([nn.Sequential(nn.Linear(D,a.cr,bias=False),nn.Linear(a.cr,D)) for _ in range(a.cl)])
            else: s.cw=nn.ModuleList([nn.Linear(D,D) for _ in range(a.cl)])
            s.cbn=nn.BatchNorm1d(D)
            if a.fmk>0: s.fv=nn.Parameter(torch.randn(D,a.fmk)*0.05); s.fbn=nn.BatchNorm1d(a.fmk)
            fin=(h//2 if a.stack else D)+(0 if a.stack else h//2)+(a.fmk if a.fmk>0 else 0)
            din=D if not a.stack else D
            s.dp=nn.Sequential(nn.Linear(D if not a.stack else D,h),nn.BatchNorm1d(h),nn.SiLU(),nn.Dropout(dr),nn.Linear(h,h//2),nn.BatchNorm1d(h//2),nn.SiLU(),nn.Dropout(dr))
            if a.stack: s.dp=nn.Sequential(nn.Linear(D,h),nn.BatchNorm1d(h),nn.SiLU(),nn.Dropout(dr),nn.Linear(h,h//2),nn.BatchNorm1d(h//2),nn.SiLU(),nn.Dropout(dr))
            s.mix=nn.Sequential(nn.Linear(fin,h//2),nn.BatchNorm1d(h//2),nn.SiLU(),nn.Dropout(dr))
            if a.stack: s.t=lambda x:s.mix(torch.cat([s.dp(s._cross(x))]+([s._fm(x)] if a.fmk>0 else []),1))
            else: s.t=lambda x:s.mix(torch.cat([s._cross(x),s.dp(x)]+([s._fm(x)] if a.fmk>0 else []),1))
        else: s.t=nn.Sequential(nn.Linear(D,h),nn.BatchNorm1d(h),nn.SiLU(),nn.Dropout(dr),nn.Linear(h,h//2),nn.BatchNorm1d(h//2),nn.SiLU(),nn.Dropout(dr))
        s.tab_o=nn.Linear(h//2,1)
        s.emb=nn.Embedding(NI+1,d,padding_idx=NI); w=np.zeros((NI+1,d),np.float32); w[:NI]=init[:,:d]; s.emb.weight.data=torch.from_numpy(w)
        s.hk=nn.Linear(d,hd); s.ht=nn.Linear(4,hd); s.null=nn.Parameter(torch.zeros(hd)); s.q=nn.Linear(d,hd)
        if a.enc=='gru': s.rnn=nn.GRU(hd,hd,batch_first=True)
        if a.enc=='trf': s.trf=nn.TransformerEncoder(nn.TransformerEncoderLayer(hd,4,2*hd,0.1,batch_first=True,norm_first=True),1)
        s.att=nn.Sequential(nn.Linear(4*hd+6,64),nn.SiLU(),nn.Linear(64,1))
        s.sp=nn.Sequential(nn.Linear(3*hd+6,hd),nn.SiLU())
        s.cv=nn.Sequential(nn.Linear(h//2+hd+hd,tk),nn.SiLU(),nn.Dropout(0.1),nn.Linear(tk,tk))
        if a.slayers>0 and not a.noslate: s.sl=nn.TransformerEncoder(nn.TransformerEncoderLayer(tk,4,2*tk,0.1,batch_first=True,norm_first=True),a.slayers)
        s.o=nn.Linear(tk,1); s.c=nn.Sequential(nn.Linear(2*tk,64),nn.SiLU(),nn.Linear(64,1))
    def _fm(s,x):
        xv=x@s.fv; return s.fbn(0.5*(xv*xv-(x*x)@(s.fv*s.fv)))
    def _cross(s,x):
        x0=x; xl=x
        for w in s.cw: xl=x0*w(xl)+xl
        return s.cbn(xl)
    def forward(s,x,cand,hi,hg,c1,c2,train=False,aug=0.0):   # x (B*10,D); cand (B*10); hi,hg (B,20); c1,c2 (B*10,20)
        B=hi.shape[0]
        if train and a.xdrop>0: x=x*(torch.rand_like(x)>a.xdrop).float()/(1-a.xdrop)
        if train and a.noise>0: x=x+a.noise*torch.randn_like(x)
        h=s.t(x); sc_tab=s.tab_o(h).squeeze(-1)
        if a.noseq:
            seq=torch.zeros(B*10,2*a.hd,device=x.device)
            if a.candid:
                ce=s.emb(cand.view(B,10))
                if train and a.edrop>0: ce=ce*(torch.rand(B,10,1)>a.edrop).float()/(1-a.edrop)
                seq=torch.cat([seq[:,:a.hd],s.q(ce).reshape(B*10,-1)],1)
        else:
            c1=c1.float().view(B,10,20); c2=c2.float().view(B,10,20)
            if train and aug>0:   # staleness augmentation: drop k (1..5) most recent clicks of a random fraction of events
                k=torch.randint(1,6,(B,1))*(torch.rand(B,1)<aug); j=torch.arange(20)[None]+k; ok=(j<20); j=j.clamp(max=19)
                hi=torch.where(ok,hi.gather(1,j),torch.full_like(hi,NI)); hg=torch.where(ok,hg.gather(1,j),torch.full_like(hg,-1.)); 
                jj=j[:,None,:].expand(B,10,20); c1=torch.where(ok[:,None],c1.gather(2,jj),torch.zeros_like(c1)); c2=torch.where(ok[:,None],c2.gather(2,jj),torch.zeros_like(c2))
            mask=(hi!=NI).float(); 
            em=s.emb(hi); 
            if train and a.edrop>0: em=em*(torch.rand(B,20,1)>a.edrop).float()/(1-a.edrop)
            H=s.hk(em)+s.ht(tf_feats(hg,mask)); H=torch.cat([s.null.expand(B,1,-1),H],1); mk=torch.cat([torch.ones(B,1),mask],1)   # (B,21,hd), null token always valid
            if a.enc=='gru': H=H+s.rnn(H.flip(1))[0].flip(1)
            if a.enc=='trf': H=s.trf(H,src_key_padding_mask=(mk==0))
            ce=s.emb(cand.view(B,10)); 
            if train and a.edrop>0: ce=ce*(torch.rand(B,10,1)>a.edrop).float()/(1-a.edrop)
            q=s.q(ce)   # (B,10,hd)
            eq=(hi[:,None,:]==cand.view(B,10,1)).float()*mask[:,None]
            tfp=tf_feats(hg,mask)[:,None,:,:3].expand(B,10,20,3)
            pf=torch.cat([c1[...,None],c2[...,None],eq[...,None],tfp],-1); pf=torch.cat([torch.zeros(B,10,1,6),pf],2)   # (B,10,21,6)
            Hh=H[:,None].expand(B,10,21,-1); qq=q[:,:,None].expand(B,10,21,-1)
            lg=s.att(torch.cat([qq,Hh,qq*Hh,qq-Hh,pf],-1)).squeeze(-1)   # (B,10,21)
            m3=mk[:,None].expand(B,10,21); a1=torch.softmax(lg.masked_fill(m3==0,-1e9),-1); o1=(a1[...,None]*Hh).sum(2)
            a2=torch.sigmoid(lg)*m3; o2=(a2[...,None]*Hh).sum(2)/(mk.sum(1)[:,None,None])
            ms=mask[:,None]; n_=ms.sum(2).clamp(min=1)
            has=(ms.sum(2)>0).float(); mx1=c1.masked_fill(ms==0,-1).max(2).values*has; mx2=c2.masked_fill(ms==0,-1).max(2).values*has
            st=torch.stack([(c1*ms).sum(2)/n_,mx1,(c2*ms).sum(2)/n_,mx2,(a1[:,:,1:]*c1).sum(2),eq.sum(2)],-1)
            seq=s.sp(torch.cat([o1,o2,q*o1,st],-1)).reshape(B*10,-1); seq=torch.cat([seq,q.reshape(B*10,-1)],1)
        z=s.cv(torch.cat([h,seq],1)).view(B,10,-1)
        if hasattr(s,'sl'): z=s.sl(z)
        sc=s.o(z).squeeze(-1).reshape(-1)+sc_tab; c=s.c(torch.cat([z.mean(1),z.max(1).values],1)).squeeze(-1)
        return sc,c
def score(sc,c): Sg=sc.view(-1,10); lse=torch.logsumexp(torch.cat([Sg,c[:,None]],1),1,keepdim=True); return (Sg-lse).reshape(-1)
def gx(A,B,j): return torch.cat([A[j],B[j].float()],1)
def run(seed):
    t1=time.time(); torch.manual_seed(seed); np.random.seed(seed); net=Net(D_,a.hid,a.drop)
    ep_=[p for n_,p in net.named_parameters() if n_=='emb.weight']; op_=[p for n_,p in net.named_parameters() if n_!='emb.weight']
    opt=torch.optim.AdamW([dict(params=op_,lr=a.lr,weight_decay=a.wd),dict(params=ep_,lr=a.elr,weight_decay=0.0)])
    bs=a.bs; steps=a.ep*((E+bs-1)//bs); sch=torch.optim.lr_scheduler.OneCycleLR(opt,[a.lr,a.elr*2],total_steps=steps,pct_start=a.pct,anneal_strategy='cos' if a.cos else 'cos'); S=[]; bce=nn.BCEWithLogitsLoss()
    for ep in range(a.ep):
        net.train(); perm=torch.randperm(E)
        for b in range(0,E,bs):
            e=perm[b:b+bs]; j=(e[:,None]*10+torch.arange(10)[None]).reshape(-1); opt.zero_grad()
            sc,c=net(gx(Xt,Ft,j),CAND[j],HI[e],HG[e],C1[j],C2[j],train=True,aug=a.aug); yy=Yt[j]
            Sg=sc.view(-1,10); Y=yy.view(-1,10); lse=torch.logsumexp(torch.cat([Sg,c[:,None]],1),1); ip=ISP[e]
            Yh=Y*(1-ip[:,None]); tgt=torch.where(Yh.sum(1)>0,(Sg*Yh).sum(1),c)
            qs_=Y.clamp(0,0.98); qs_=qs_/qs_.sum(1,keepdim=True).clamp(min=0.98); qo=(1-qs_.sum(1)).clamp(min=0)
            tgt=torch.where(ip>0,(Sg*qs_).sum(1)+qo*c,tgt); w_=1+(a.tw-1)*ip
            bl=nn.functional.binary_cross_entropy_with_logits(sc,(yy.clamp(0,1)*(1-a.ls)+a.ls*0.06),reduction='none').view(-1,10).mean(1)
            l=a.lw*((lse-tgt)*w_).mean()+(a.alpha*(bl*w_).mean() if a.alpha>0 else 0); l.backward(); nn.utils.clip_grad_norm_(net.parameters(),5.0); opt.step(); sch.step()
        net.eval(); P=[]
        with torch.no_grad():
            for b in range(0,Ev,4000):
                e=torch.arange(b,min(b+4000,Ev)); j=(e[:,None]*10+torch.arange(10)[None]).reshape(-1)
                sc,c=net(gx(Xvt,Fvt,j),CANDv[j],HIv[e],HGv[e],C1v[j],C2v[j]); P.append(score(sc,c))
        p=torch.cat(P).numpy(); S.append(p); msg=f'seed{seed} ep{ep} {time.time()-t0:.0f}s'
        if a.cut!=27: msg+=f' auc {roc_auc_score(mv["clicked"].to_numpy(),p):.5f}'
        print(msg,flush=True)
    np.save(f'{a.outdir}/{a.tag}_{a.var}_c{a.cut}_s{seed}.npy',np.stack(S,1).astype(np.float32))
    open('/workspace/huawei/v21dcn/logs/dcn_log.jsonl','a').write(json.dumps(dict(cut=a.cut,tag=a.tag,var=a.var,seed=seed,args=vars(a),min=round((time.time()-t1)/60,1),aucs=[float(roc_auc_score(mv['clicked'].to_numpy(),s_)) for s_ in S] if a.cut!=27 else None))+'\n')
if not a.cfg:
    for sd in [int(x) for x in a.seeds.split(',')]: run(sd)
else:
    import copy; base=copy.deepcopy(a)
    for c in json.load(open(a.cfg)):
        for k,v in vars(base).items(): setattr(a,k,v)
        a.tag=c['name']
        for k,v in c.items():
            if k not in('name','seeds'): setattr(a,k,v)
        print('CFG',c,flush=True)
        for sd in c.get('seeds',[0]): run(sd)
