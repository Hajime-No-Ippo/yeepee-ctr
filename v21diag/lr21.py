"""mlp12.py CUT TAG [--ep 4] [--seed 0]: MLP on the same LightGBM matrix (train f(C-24)..f(C-21), negfrac .5, v10B drop), rank-gauss inputs
from TRAIN quantiles only + missing flags. Saves ens/{TAG}_c{CUT}_s{seed}.npy (n,EP) logits after each epoch."""
import os, sys; sys.path.insert(0,'/workspace/huawei/v19'); os.chdir('/workspace/huawei/v21diag')
import re, time, argparse, json, gc, numpy as np, torch, torch.nn as nn
from cl import matrix
from sklearn.metrics import roc_auc_score
ap=argparse.ArgumentParser(); ap.add_argument('cut',type=int); ap.add_argument('tag'); ap.add_argument('--ep',type=int,default=4); ap.add_argument('--seed',type=int,default=0)
ap.add_argument('--lr',default='feat_lr2,feat_tt,feat_pp,feat_sqC,feat_dw'); ap.add_argument('--loss',default='list'); ap.add_argument('--alpha',type=float,default=0.0); ap.add_argument('--hid',type=int,default=512); ap.add_argument('--wd',type=float,default=1e-2); ap.add_argument('--drop',type=float,default=0.2)
a=ap.parse_args(); t0=time.time(); torch.manual_seed(a.seed); np.random.seed(a.seed); torch.set_num_threads(int(os.environ.get('MLP_THREADS',8)))
DR='^sq_(pdot|pcos|mdot|mcos)$'; extra=tuple(a.lr.split(','))
trn=[f'f{k}' for k in range(max(1,a.cut-24),a.cut-20)]
rs=np.random.RandomState(123); Xs=[];ys=[]
for n in trn:
    X,m,f=matrix(n,1.0,extra=extra); y=m['clicked'].to_numpy(); assert len(y)%10==0 and (m['eid'].to_numpy().reshape(-1,10)==m['eid'].to_numpy().reshape(-1,10)[:,:1]).all(); feats=f; keep=[i for i,c in enumerate(feats) if not re.search(DR,c)]; Xs.append(np.ascontiguousarray(X[:,keep])); ys.append(y); del X; gc.collect()
fl=[feats[i] for i in keep]
NR=sum(len(a_) for a_ in Xs); X=np.empty((NR,Xs[0].shape[1]),np.float32); o_=0
while Xs:
    a_=Xs.pop(0); X[o_:o_+len(a_)]=a_; o_+=len(a_); del a_; gc.collect()
y=np.concatenate(ys).astype(np.float32); gc.collect()
vn='eval' if a.cut==27 else f'val{a.cut}'; Xv,mv,fv=matrix(vn,1.0,extra=extra); Xv=Xv[:,[fv.index(c) for c in fl]]
# rank-gauss via train quantiles (101 knots); NaN -> 0 + flag
from scipy.special import ndtri
qs=np.linspace(0.001,0.999,101); Q=np.nanquantile(X[::7],qs,axis=0).astype(np.float32)   # (101,F)
nanc=np.where(np.isnan(X[::7]).any(0)|np.isnan(Xv[::7]).any(0))[0]; G=ndtri(qs).astype(np.float32)
def tr(Z):
    F=Z.shape[1]; fl_=np.isnan(Z[:,nanc]); 
    for j in range(F):
        q=np.nan_to_num(Q[:,j],nan=0.0); v=Z[:,j]; u,ix=np.unique(q,return_index=True)
        nn_=np.isnan(v); o=np.interp(np.nan_to_num(v),u,G[ix]) if len(u)>1 else np.zeros_like(v); o[nn_]=0; Z[:,j]=o
    return Z,fl_
X,FX=tr(X); gc.collect(); Xv,FV=tr(Xv); gc.collect(); print('prep',X.shape,FX.shape,round(time.time()-t0),'s',flush=True)
D=X.shape[1]+len(nanc)
class Lin(nn.Module):
    def __init__(s,D):
        super().__init__(); s.l=nn.Linear(D,1)
    def forward(s,x): return s.l(x).squeeze(-1), torch.zeros(len(x)//10)
class Net(nn.Module):
    def __init__(s,D,h,dr):
        super().__init__()
        s.t=nn.Sequential(nn.Linear(D,h),nn.BatchNorm1d(h),nn.SiLU(),nn.Dropout(dr),nn.Linear(h,h//2),nn.BatchNorm1d(h//2),nn.SiLU(),nn.Dropout(dr))
        s.o=nn.Linear(h//2,1); s.c=nn.Sequential(nn.Linear(2*(h//2),64),nn.SiLU(),nn.Linear(64,1))
    def forward(s,x):   # x (B*10,D) slate-contiguous
        h=s.t(x); sc=s.o(h).squeeze(-1); H=h.view(-1,10,h.shape[1]); c=s.c(torch.cat([H.mean(1),H.max(1).values],1)).squeeze(-1)
        return sc,c
net=Lin(D); opt=torch.optim.AdamW(net.parameters(),1e-3,weight_decay=a.wd); bce=nn.BCEWithLogitsLoss()
Xt=torch.from_numpy(X); Ft=torch.from_numpy(FX); Yt=torch.from_numpy(y); Xvt=torch.from_numpy(Xv); Fvt=torch.from_numpy(FV);
def gx(A,B,j): return torch.cat([A[j],B[j].float()],1)
E=len(y)//10
bs=256; steps=a.ep*((E+bs-1)//bs); sch=torch.optim.lr_scheduler.OneCycleLR(opt,2e-3,total_steps=steps,pct_start=0.1); S=[]
def score(sc,c):   # log p_i of the conditional logit with outside option c (logit of "no click")
    Sg=sc.view(-1,10); lse=torch.logsumexp(torch.cat([Sg,c[:,None]],1),1,keepdim=True); return (Sg-lse).reshape(-1)
for ep in range(a.ep):
    net.train(); perm=torch.randperm(E)
    for b in range(0,E,bs):
        e=perm[b:b+bs]; j=(e[:,None]*10+torch.arange(10)[None]).reshape(-1); opt.zero_grad(); sc,c=net(gx(Xt,Ft,j)); yy=Yt[j]
        if a.loss=='list':
            Sg=sc.view(-1,10); Y=yy.view(-1,10); lse=torch.logsumexp(torch.cat([Sg,c[:,None]],1),1)
            tgt=torch.where(Y.sum(1)>0,(Sg*Y).sum(1),c); l=(lse-tgt).mean()+a.alpha*bce(sc,yy)
        else: l=bce(sc,yy)
        l.backward(); opt.step(); sch.step()
    net.eval()
    with torch.no_grad():
        outs=[net(gx(Xvt,Fvt,slice(b,b+65000))) for b in range(0,len(Xv),65000)]   # 65000 divisible by 10
        p=torch.cat([score(o[0],o[1]) if a.loss=='list' else o[0] for o in outs]).numpy()
    S.append(p); msg=f'ep{ep} {time.time()-t0:.0f}s'
    if a.cut!=27: msg+=f' auc {roc_auc_score(mv["clicked"].to_numpy(),p):.5f}'
    print(msg,flush=True)
os.makedirs('ens',exist_ok=True); np.save(f'ens/{a.tag}_c{a.cut}_s{a.seed}.npy',np.stack(S,1).astype(np.float32))
open('/workspace/huawei/v21diag/logs/lr21_log.jsonl','a').write(json.dumps(dict(cut=a.cut,tag=a.tag,seed=a.seed,min=round((time.time()-t0)/60,1)))+'\n')
