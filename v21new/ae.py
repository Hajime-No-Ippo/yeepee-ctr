"""History-only (ID-free user) next-click autoencoder: user vector = f(last<=20 clicks); sampled-softmax over items w/ logQ correction.
Train data: train-only click chronology at cutoff X_k (graph file g_k: cu,ci,ct). Score = logit(item | event history).
python ae.py K TAG [--d 256 --ep 12 --seed 0 --mon val25 --drop .3]"""
import warnings; warnings.filterwarnings('ignore')
import sys,time,argparse,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
from sklearn.metrics import roc_auc_score
sys.dont_write_bytecode=True
ap=argparse.ArgumentParser(); ap.add_argument('k',type=int); ap.add_argument('tag'); ap.add_argument('--d',type=int,default=256); ap.add_argument('--ep',type=int,default=12)
ap.add_argument('--seed',type=int,default=0); ap.add_argument('--mon',default=''); ap.add_argument('--drop',type=float,default=0.3); ap.add_argument('--lr',type=float,default=2e-3)
ap.add_argument('--bs',type=int,default=1024); ap.add_argument('--nneg',type=int,default=8192); ap.add_argument('--wd',type=float,default=0.0); ap.add_argument('--thr',type=int,default=4)
ap.add_argument('--save',default=''); ap.add_argument('--L',type=int,default=20); ap.add_argument('--hid',type=int,default=1); ap.add_argument('--tie',type=int,default=0)
a=ap.parse_args(); torch.set_num_threads(a.thr); torch.manual_seed(a.seed); np.random.seed(a.seed)
G='/workspace/huawei/v21graph2/out/'; O='/workspace/huawei/v21new/out/'; NI=50000; PAD=NI; L=a.L; t0=time.time()
g=np.load(G+f'g_{a.k}.npz'); cu,ci,ct=g['cu'].astype(np.int64),g['ci'].astype(np.int64),g['ct']
o=np.lexsort((ci,ct,cu)); cu,ci,ct=cu[o],ci[o],ct[o]
keep=np.ones(len(cu),bool); keep[1:]=~((cu[1:]==cu[:-1])&(ci[1:]==ci[:-1])&(ct[1:]==ct[:-1])); cu,ci,ct=cu[keep],ci[keep],ct[keep]
n=len(cu); start=np.r_[0,np.flatnonzero(cu[1:]!=cu[:-1])+1]; ulen=np.diff(np.r_[start,n]); first=np.repeat(start,ulen); pos=np.arange(n)-first
# history matrix: previous up to L clicks newest first
Hm=np.full((n,L),PAD,np.int32)
for j in range(1,L+1):
    ok=pos>=j; idx=np.flatnonzero(ok); Hm[idx,j-1]=ci[idx-j]
m=pos>=1; X=torch.from_numpy(Hm[m]); Y=torch.from_numpy(ci[m]); print('samples',len(Y),flush=True)
cnt=np.bincount(ci,minlength=NI).astype(np.float64)+1; q=cnt**0.75; q/=q.sum(); logq=torch.from_numpy(np.log(q)).float(); Q=torch.from_numpy(q)
class Net(nn.Module):
    def __init__(s,d):
        super().__init__(); s.E=nn.Embedding(NI+1,d,padding_idx=PAD); s.pw=nn.Parameter(torch.zeros(L)); s.O=s.E if a.tie else nn.Embedding(NI,d); s.b=nn.Parameter(torch.zeros(NI))
        nn.init.normal_(s.E.weight,0,0.05); (None if a.tie else nn.init.normal_(s.O.weight,0,0.05))
        s.ln=nn.LayerNorm(d); s.f=nn.ModuleList([nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Dropout(0.1)) for _ in range(a.hid)])
    def user(s,x):
        w=torch.exp(s.pw)[None,:,None]*(x!=PAD).unsqueeze(-1); h=(s.E(x)*w).sum(1)/torch.sqrt((x!=PAD).sum(1,keepdim=True).clamp(min=1).float())
        h=s.ln(h)
        for f in s.f: h=h+f(h)
        return h
net=Net(a.d); opt=torch.optim.AdamW(net.parameters(),lr=a.lr,weight_decay=a.wd)
mon=None
def load(k):
    T=np.load(G+f'tab_{k}.npz'); return T
def score(T):
    hist=torch.from_numpy(T['hist'][:,:L].astype(np.int64)); item=torch.from_numpy(T['item'].astype(np.int64)).reshape(-1,10)
    net.eval(); out=[]
    with torch.no_grad():
        for i in range(0,len(hist),8192):
            u=net.user(hist[i:i+8192]); it=item[i:i+8192]; s=(u[:,None,:]*net.O(it)).sum(-1)+net.b[it]; out.append(s.numpy())
    net.train(); return np.concatenate(out).reshape(-1)
if a.mon: Tm=load(a.mon)
def mon_eval(ep):
    s=score(Tm); y=Tm['y']; sd=s-s.reshape(-1,10).mean(1,keepdims=True).repeat(10,1).reshape(-1)
    print(f'ep {ep} {time.time()-t0:.0f}s {a.mon} auc raw {roc_auc_score(y,s):.5f} demeaned {roc_auc_score(y,sd):.5f}',flush=True)
N=len(Y); saves=[int(x) for x in a.save.split(',') if x]
import time as _t
for ep in range(1,a.ep+1):
    perm=torch.randperm(N); tl=0;nb=0
    for i in range(0,N-a.bs+1,a.bs):
        b=perm[i:i+a.bs]; x=X[b].clone(); y=Y[b]
        if a.drop>0: x[torch.rand(x.shape)<a.drop]=PAD
        u=net.user(x); neg=torch.multinomial(Q,a.nneg,replacement=True)
        ap_=(u*net.O(y)).sum(-1)+net.b[y]-logq[y]; an=u@net.O(neg).T+net.b[neg][None,:]-logq[neg][None,:]
        an=an.masked_fill(neg[None,:]==y[:,None],-1e9)
        loss=F.cross_entropy(torch.cat([ap_[:,None],an],1),torch.zeros(len(y),dtype=torch.long))
        opt.zero_grad(); loss.backward(); opt.step(); tl+=loss.item(); nb+=1
        if nb%50==0 and ep==1: print('step',nb,f'{time.time()-t0:.0f}s',flush=True)
    print(f'ep {ep} loss {tl/nb:.4f} {time.time()-t0:.0f}s',flush=True)
    if a.mon: mon_eval(ep)
    if ep in saves:
        import os; os.makedirs(O,exist_ok=True); np.savez(O+f'ae_{a.tag}_k{a.k}_s{a.seed}_e{ep}.npz',**{f'{k}':np.float32(0) for k in []})
        for nm in ['val23','val25','val26','eval']:
            pass
