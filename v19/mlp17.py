"""mlp12.py CUT TAG [--ep 4] [--seed 0]: MLP on the same LightGBM matrix (train f(C-24)..f(C-21), negfrac .5, v10B drop), rank-gauss inputs
from TRAIN quantiles only + missing flags. Saves ens/{TAG}_c{CUT}_s{seed}.npy (n,EP) logits after each epoch."""
import os, sys; sys.path.insert(0,'/workspace/huawei/v19')
import re, time, argparse, json, gc, numpy as np, torch, torch.nn as nn
from cl import matrix
from sklearn.metrics import roc_auc_score
ap=argparse.ArgumentParser(); ap.add_argument('cut',type=int); ap.add_argument('tag'); ap.add_argument('--ep',type=int,default=4); ap.add_argument('--seed',type=int,default=0)
ap.add_argument('--lr',default='feat_lr2,feat_tt,feat_pp,feat_sqC,feat_dw'); ap.add_argument('--hid',type=int,default=512); ap.add_argument('--drop',type=float,default=0.2)
a=ap.parse_args(); t0=time.time(); torch.manual_seed(a.seed); np.random.seed(a.seed); torch.set_num_threads(int(os.environ.get('MLP_THREADS',8)))
DR='^sq_(pdot|pcos|mdot|mcos)$'; extra=tuple(a.lr.split(','))
trn=[f'f{k}' for k in range(max(1,a.cut-24),a.cut-20)]
rs=np.random.RandomState(123); Xs=[];ys=[]
for n in trn:
    X,m,f=matrix(n,1.0,extra=extra); y=m['clicked'].to_numpy(); sel=(y==1)|(rs.rand(len(y))<0.5); Xs.append(X[sel]); ys.append(y[sel]); feats=f; del X; gc.collect()
keep=[i for i,c in enumerate(feats) if not re.search(DR,c)]; fl=[feats[i] for i in keep]
X=np.concatenate([x[:,keep] for x in Xs]); del Xs; y=np.concatenate(ys).astype(np.float32); gc.collect()
vn='eval' if a.cut==27 else f'val{a.cut}'; Xv,mv,fv=matrix(vn,1.0,extra=extra); Xv=Xv[:,[fv.index(c) for c in fl]]
# rank-gauss via train quantiles (101 knots); NaN -> 0 + flag
from scipy.special import ndtri
qs=np.linspace(0.001,0.999,101); Q=np.nanquantile(X[::7],qs,axis=0).astype(np.float32)   # (101,F)
nanc=np.where(np.isnan(X[::7]).any(0)|np.isnan(Xv[::7]).any(0))[0]; G=ndtri(qs).astype(np.float32)
def tr(Z):
    out=np.empty((len(Z),Z.shape[1]+len(nanc)),np.float32)
    for j in range(Z.shape[1]):
        q=np.nan_to_num(Q[:,j],nan=0.0); v=Z[:,j]; u,ix=np.unique(q,return_index=True)
        o=np.interp(v,u,G[ix]) if len(u)>1 else np.zeros_like(v); o[np.isnan(v)]=0; out[:,j]=o
    out[:,Z.shape[1]:]=np.isnan(Z[:,nanc]); return out
X=np.nan_to_num(tr(X)); Xv=np.nan_to_num(tr(Xv)); gc.collect(); print('prep',X.shape,round(time.time()-t0),'s',flush=True)
D=X.shape[1]
net=nn.Sequential(nn.Linear(D,a.hid),nn.BatchNorm1d(a.hid),nn.SiLU(),nn.Dropout(a.drop),nn.Linear(a.hid,a.hid//2),nn.BatchNorm1d(a.hid//2),nn.SiLU(),nn.Dropout(a.drop),nn.Linear(a.hid//2,1))
opt=torch.optim.AdamW(net.parameters(),1e-3,weight_decay=1e-4); lf=nn.BCEWithLogitsLoss(); Xt=torch.from_numpy(X); Yt=torch.from_numpy(y); Xvt=torch.from_numpy(Xv)
bs=2048; steps=a.ep*((len(y)+bs-1)//bs); sch=torch.optim.lr_scheduler.OneCycleLR(opt,2e-3,total_steps=steps,pct_start=0.1); S=[]
for ep in range(a.ep):
    net.train(); perm=torch.randperm(len(y))
    for b in range(0,len(y),bs):
        j=perm[b:b+bs]; opt.zero_grad(); l=lf(net(Xt[j]).squeeze(-1),Yt[j]); l.backward(); opt.step(); sch.step()
    net.eval()
    with torch.no_grad(): p=torch.cat([net(Xvt[b:b+65536]).squeeze(-1) for b in range(0,len(Xv),65536)]).numpy()
    S.append(p); msg=f'ep{ep} {time.time()-t0:.0f}s'
    if a.cut!=27: msg+=f' auc {roc_auc_score(mv["clicked"].to_numpy(),p):.5f}'
    print(msg,flush=True)
os.makedirs('ens',exist_ok=True); np.save(f'ens/{a.tag}_c{a.cut}_s{a.seed}.npy',np.stack(S,1).astype(np.float32))
open('ens/ens_log.jsonl','a').write(json.dumps(dict(cut=a.cut,tag=a.tag,seed=a.seed,min=round((time.time()-t0)/60,1)))+'\n')
