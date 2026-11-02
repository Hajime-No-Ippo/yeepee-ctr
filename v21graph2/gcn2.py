"""LightGCN (+slate softmax on logged events) on the TRAIN-ONLY click graph at cutoff X_k.
python gcn.py K TAG [--d 32 --L 2 --l2 1e-4 --steps 600 --lr 5e-3 --seed 0 --mon val25 --every 100 --slate 1.0 --bpr 1.0]
Saves out/emb_{TAG}_k{K}_s{seed}.npz (U,V final-propagated at each checkpoint step list in --save) ."""
import warnings; warnings.filterwarnings('ignore')
import sys,os,time,argparse,numpy as np,torch; sys.dont_write_bytecode=True
from sklearn.metrics import roc_auc_score
ap=argparse.ArgumentParser(); ap.add_argument('k',type=int); ap.add_argument('tag'); ap.add_argument('--d',type=int,default=32); ap.add_argument('--L',type=int,default=2)
ap.add_argument('--l2',type=float,default=1e-4); ap.add_argument('--steps',type=int,default=600); ap.add_argument('--lr',type=float,default=5e-3); ap.add_argument('--seed',type=int,default=0)
ap.add_argument('--mon',default=''); ap.add_argument('--every',type=int,default=100); ap.add_argument('--slate',type=float,default=1.0); ap.add_argument('--bpr',type=float,default=1.0)
ap.add_argument('--bs',type=int,default=16384); ap.add_argument('--drop',type=float,default=0.0); ap.add_argument('--save',default=''); ap.add_argument('--thr',type=int,default=2); ap.add_argument('--tau',type=float,default=0); ap.add_argument('--stau',type=float,default=0)
a=ap.parse_args(); torch.set_num_threads(a.thr); O='/workspace/huawei/v21graph2/out/'; t0=time.time()
NU,NI=np.load(O+'dims.npy'); g=np.load(O+f'g_{a.k}.npz'); cu,ci=g['cu'].astype(np.int64),g['ci'].astype(np.int64); nX=int(g['nX'])
key,inv=np.unique(cu*NI+ci,return_inverse=True); eu=key//NI; ei=key%NI; E_=len(key)
ets=np.zeros(E_,np.int64); np.maximum.at(ets,inv,g['ct']); Xt=int(g['ct'].max())+1
ewt=torch.from_numpy(np.exp(-(Xt-ets)/(86400.*a.tau)) if a.tau>0 else np.ones(E_)).float()
# logged slates (train events with ts<X): need D arrays -> cached by prep as slates.npz
sl=np.load(O+'slates.npz'); keep=sl['e']<nX; SU=torch.from_numpy(sl['u'][keep].astype(np.int64)); SC=torch.from_numpy(sl['cand'][keep].astype(np.int64)); SY=torch.from_numpy(sl['pos'][keep].astype(np.int64))
print('edges',E_,'slate events',len(SU),flush=True)
def adj(eu_,ei_):
    r=np.concatenate([eu_,NU+ei_]); c=np.concatenate([NU+ei_,eu_]); v=np.ones(len(r),np.float32)
    deg=np.bincount(r,minlength=NU+NI).astype(np.float32); v=v/np.sqrt(deg[r]*deg[c])
    return torch.sparse_coo_tensor(torch.from_numpy(np.stack([r,c])),torch.from_numpy(v),(NU+NI,NU+NI)).coalesce().to_sparse_csr()
A=adj(eu,ei); torch.manual_seed(a.seed); np.random.seed(a.seed)
emb=torch.nn.Parameter(torch.randn(NU+NI,a.d)*0.1); opt=torch.optim.Adam([emb],lr=a.lr)
class SpMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx,Am,x): ctx.Am=Am; return Am@x
    @staticmethod
    def backward(ctx,g): return None,ctx.Am@g   # symmetric adjacency
def prop(x,Am):
    outs=[x]; h=x
    for _ in range(a.L): h=SpMM.apply(Am,h); outs.append(h)
    return torch.stack(outs).mean(0)
EU=torch.from_numpy(eu); EI=torch.from_numpy(ei)
pop=np.bincount(ei,minlength=NI).astype(np.float64)**0.75; pop=torch.from_numpy(pop/pop.sum())
mon=None
if a.mon:
    T=np.load(O+f'tab_{a.mon}.npz'); mu=torch.from_numpy(T['user'].astype(np.int64)); mi=torch.from_numpy(T['item'].astype(np.int64)); my=T['y']
    mu=mu.clamp(max=NU-1); mon=True
def monitor(step):
    with torch.no_grad(): z=prop(emb,A); U=z[:NU]; V=z[NU:]; s=(U[mu]*V[mi]).sum(1).numpy()
    sd=s-s.reshape(-1,10).mean(1,keepdims=True).repeat(10,1).reshape(-1)
    print(f'step {step} {time.time()-t0:.0f}s {a.mon} auc raw {roc_auc_score(my,s):.5f} demeaned {roc_auc_score(my,sd):.5f}',flush=True)
saves=[int(x) for x in a.save.split(',') if x]; res={}
for step in range(1,a.steps+1):
    opt.zero_grad()
    if a.drop>0:
        m=torch.rand(E_)>a.drop; Ad=adj(eu[m.numpy()],ei[m.numpy()])
    else: Ad=A
    z=prop(emb,Ad); U=z[:NU]; V=z[NU:]; loss=0.
    if a.bpr>0:
        b=torch.multinomial(ewt,a.bs,replacement=True) if a.tau>0 else torch.randint(0,E_,(a.bs,)); u=EU[b]; p=EI[b]; n=torch.multinomial(pop,a.bs,replacement=True)
        loss=loss+a.bpr*torch.nn.functional.softplus((U[u]*V[n]).sum(1)-(U[u]*V[p]).sum(1)).mean()
    if a.slate>0:
        b=torch.randint(0,len(SU),(a.bs//2,)); u=SU[b]; c=SC[b]; y=SY[b]   # y in 0..9 (event has a click) else skipped
        ok=y>=0; u,c,y=u[ok],c[ok],y[ok]
        sc=(U[u][:,None,:]*V[c]).sum(-1); loss=loss+a.slate*torch.nn.functional.cross_entropy(sc,y)
    reg=a.l2*(emb[:NU].pow(2).sum(1).mean()+emb[NU:].pow(2).sum(1).mean()); (loss+reg).backward(); opt.step()
    if mon is not None and step%a.every==0: monitor(step)
    if step in saves:
        with torch.no_grad(): z=prop(emb,A); res[f'U{step}']=z[:NU].numpy().astype(np.float32); res[f'V{step}']=z[NU:].numpy().astype(np.float32)
if res: np.savez(O+f'emb_{a.tag}_k{a.k}_s{a.seed}.npz',**res)
print('done',time.time()-t0)
