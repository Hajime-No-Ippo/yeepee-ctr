"""Set-autoencoder (Mult-DAE style, leave-one-out): input = user's other clicks at cutoff X_k (train-only graph) (+event history at inference),
target = held-out click; tied item embeddings; sampled softmax w/ logQ.  python ae2.py K TAG [opts]"""
import warnings; warnings.filterwarnings('ignore')
import sys,time,argparse,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
from sklearn.metrics import roc_auc_score
sys.dont_write_bytecode=True
ap=argparse.ArgumentParser(); ap.add_argument('k',type=int); ap.add_argument('tag'); ap.add_argument('--d',type=int,default=64); ap.add_argument('--ep',type=int,default=12)
ap.add_argument('--seed',type=int,default=0); ap.add_argument('--mon',default=''); ap.add_argument('--drop',type=float,default=0.3); ap.add_argument('--lr',type=float,default=4e-3)
ap.add_argument('--bs',type=int,default=4096); ap.add_argument('--nneg',type=int,default=8192); ap.add_argument('--wd',type=float,default=0.0); ap.add_argument('--thr',type=int,default=2)
ap.add_argument('--hid',type=int,default=1); ap.add_argument('--tie',type=int,default=1); ap.add_argument('--cap',type=int,default=60); ap.add_argument('--evalsave',default='')
a=ap.parse_args(); torch.set_num_threads(a.thr); torch.manual_seed(a.seed); np.random.seed(a.seed)
G='/workspace/huawei/v21graph2/out/'; O='/workspace/huawei/v21new/out/'; NI=50000; PAD=NI; W=a.cap; t0=time.time()
g=np.load(G+f'g_{a.k}.npz'); cu,ci,ct=g['cu'].astype(np.int64),g['ci'].astype(np.int64),g['ct']
o=np.lexsort((-ct,cu)); cu,ci=cu[o],ci[o]  # per user newest first
key=cu*NI+ci; _,fi=np.unique(key,return_index=True); fi=np.sort(fi); cu,ci=cu[fi],ci[fi]   # unique (u,i) keeping first (newest) occurrence order
NU=int(np.load(G+'dims.npy')[0]); n=len(cu); start=np.r_[0,np.flatnonzero(cu[1:]!=cu[:-1])+1]; ulen=np.diff(np.r_[start,n]); pos=np.arange(n)-np.repeat(start,ulen)
UC=np.full((NU+1,W),PAD,np.int32); m=pos<W; UC[cu[m],pos[m]]=ci[m]; UCt=torch.from_numpy(UC.astype(np.int64))
# training pairs: all (u,i) edges with user having >=2 clicks
ulen_u=np.zeros(NU+1,int); ulen_u[cu[start]]=ulen; ok=ulen_u[cu]>=2; TU=torch.from_numpy(cu[ok]); TI=torch.from_numpy(ci[ok].astype(np.int64)); N=len(TU); print('pairs',N,'users',len(start),flush=True)
cnt=np.bincount(ci,minlength=NI).astype(np.float64)+1; q=cnt**0.75; q/=q.sum(); logq=torch.from_numpy(np.log(q)).float(); Q=torch.from_numpy(q)
class Net(nn.Module):
    def __init__(s,d):
        super().__init__(); s.E=nn.Embedding(NI+1,d,padding_idx=PAD); s.O=s.E if a.tie else nn.Embedding(NI,d); s.b=nn.Parameter(torch.zeros(NI))
        nn.init.normal_(s.E.weight,0,0.05)
        if not a.tie: nn.init.normal_(s.O.weight,0,0.05)
        s.ln=nn.LayerNorm(d); s.f=nn.ModuleList([nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Dropout(0.1)) for _ in range(a.hid)])
    def user(s,x):
        msk=(x!=PAD); h=s.E(x).sum(1)/torch.sqrt(msk.sum(1,keepdim=True).clamp(min=1).float()); h=s.ln(h)
        for f in s.f: h=h+f(h)
        return h
net=Net(a.d); opt=torch.optim.AdamW(net.parameters(),lr=a.lr,weight_decay=a.wd)
def build_in(T):
    hist=T['hist'].astype(np.int64); us=np.minimum(T['user'].reshape(-1,10)[:,0].astype(np.int64),NU)
    x=np.concatenate([hist,UC[us].astype(np.int64)],1); x.sort(1); dup=np.zeros_like(x,bool); dup[:,1:]=x[:,1:]==x[:,:-1]; x[dup]=PAD; return torch.from_numpy(x)
def score(T):
    X=build_in(T); item=torch.from_numpy(T['item'].astype(np.int64)).reshape(-1,10); net.eval(); out=[]
    with torch.no_grad():
        for i in range(0,len(X),4096):
            u=net.user(X[i:i+4096]); it=item[i:i+4096]; out.append(((u[:,None,:]*net.O(it)).sum(-1)+net.b[it]).numpy())
    net.train(); return np.concatenate(out).reshape(-1)
if a.mon: Tm=np.load(G+f'tab_{a.mon}.npz')
def mon_eval(ep):
    s=score(Tm); y=Tm['y']; sd=s-s.reshape(-1,10).mean(1,keepdims=True).repeat(10,1).reshape(-1)
    print(f'ep {ep} {time.time()-t0:.0f}s {a.mon} auc raw {roc_auc_score(y,s):.5f} demeaned {roc_auc_score(y,sd):.5f}',flush=True)
for ep in range(1,a.ep+1):
    perm=torch.randperm(N); tl=0;nb=0
    for i in range(0,N-a.bs+1,a.bs):
        b=perm[i:i+a.bs]; y=TI[b]; x=UCt[TU[b]].clone(); x[x==y[:,None]]=PAD
        if a.drop>0: x[torch.rand(x.shape)<a.drop]=PAD
        u=net.user(x); neg=torch.multinomial(Q,a.nneg,replacement=True)
        ap_=(u*net.O(y)).sum(-1)+net.b[y]-logq[y]; an=u@net.O(neg).T+net.b[neg][None,:]-logq[neg][None,:]
        an=an.masked_fill(neg[None,:]==y[:,None],-1e9)
        loss=F.cross_entropy(torch.cat([ap_[:,None],an],1),torch.zeros(len(y),dtype=torch.long))
        opt.zero_grad(); loss.backward(); opt.step(); tl+=loss.item(); nb+=1
    print(f'ep {ep} loss {tl/nb:.4f} {time.time()-t0:.0f}s',flush=True)
    if a.mon: mon_eval(ep)
    if str(ep) in a.evalsave.split(','):
        import os; os.makedirs(O,exist_ok=True)
        for nm in {3:'val23',5:'val25',6:'val26',7:'eval'}[a.k].split(','):
            np.save(O+f'ae2_{a.tag}_{nm}_s{a.seed}_e{ep}.npy',score(np.load(G+f'tab_{nm}.npz')).astype(np.float32))
print('done',time.time()-t0)
