# softmax-within-event two-tower with outside option; trained on pre-J25 train events (all users)
import numpy as np, torch, sys, polars as pl, argparse
torch.set_num_threads(2); pass
ap=argparse.ArgumentParser(); ap.add_argument('--d',type=int,default=12); ap.add_argument('--ep',type=int,default=6)
ap.add_argument('--user',type=int,default=0); ap.add_argument('--tied',type=int,default=0); ap.add_argument('--nonull',type=int,default=0)
ap.add_argument('--init',default='');ap.add_argument('--freeze',type=int,default=0);ap.add_argument('--lr',type=float,default=0.005)
ap.add_argument('--nobias',type=int,default=0)
ap.add_argument('--wd',type=float,default=0.0); ap.add_argument('--tag',default='m'); ap.add_argument('--nullu',type=int,default=1)
ap.add_argument('--seed',type=int,default=0)
a=ap.parse_args(); torch.manual_seed(a.seed); np.random.seed(a.seed)
T=np.load('/tmp/dig/tr_ev.npz'); V=np.load('/tmp/dig/val_t.npz')
NI=50000; NU=int(T['uid'].max())+1; d=a.d
# map val users to train uid
um=dict(zip(T['user'],T['uid'])); vu=np.array([um.get(u,-1) for u in V['evuser']])
class M(torch.nn.Module):
    def __init__(s):
        super().__init__()
        s.P=torch.nn.Embedding(NI+1,d,padding_idx=NI); s.Q=s.P if a.tied else torch.nn.Embedding(NI,d)
        s.b=torch.nn.Embedding(NI,1); s.c0=torch.nn.Parameter(torch.zeros(1)); s.wn=torch.nn.Linear(d,1)
        torch.nn.init.normal_(s.P.weight,0,0.1); 
        if not a.tied: torch.nn.init.normal_(s.Q.weight,0,0.1)
        torch.nn.init.zeros_(s.b.weight)
        if a.user: s.E=torch.nn.Embedding(NU+1,d,padding_idx=NU); torch.nn.init.zeros_(s.E.weight); s.ub=torch.nn.Embedding(NU+1,1,padding_idx=NU); torch.nn.init.zeros_(s.ub.weight)
        with torch.no_grad(): s.P.weight[NI]=0
    def uvec(s,H,uid):
        m=(H<NI).float(); u=(s.P(H)*m[...,None]).sum(1)/m.sum(1,keepdim=True).clamp(min=1)
        if a.user: u=u+s.E(uid)
        return u
    def forward(s,H,cand,uid):
        u=s.uvec(H,uid); sc=(u[:,None,:]*s.Q(cand)).sum(-1)*(s.scale if hasattr(s,'scale') else 1)+s.b(cand)[...,0]
        null=s.c0+(s.wn(u)[:,0] if a.nullu else 0)
        if a.user: null=null+s.ub(uid)[:,0]
        return sc,null
m=M()
if a.init:
    W=np.load(a.init).astype(np.float32); W=W/(np.linalg.norm(W,axis=1,keepdims=True)+1e-9); W=W[:,:d]
    with torch.no_grad():
        m.P.weight[:NI]=torch.from_numpy(W)
        if not a.tied: m.Q.weight[:]=torch.from_numpy(W)
    if a.freeze:
        m.P.weight.requires_grad_(False); 
        if not a.tied: m.Q.weight.requires_grad_(False)
        m.scale=torch.nn.Parameter(torch.ones(1))
Ht=torch.from_numpy(np.where(T['H']<0,NI,T['H']).astype(np.int64)); Ct=torch.from_numpy(T['cand'].astype(np.int64)); Lt=torch.from_numpy(T['lab'].astype(np.int64)); Ut=torch.from_numpy(T['uid'].astype(np.int64))
Hv=torch.from_numpy(np.where(V['H']<0,NI,V['H']).astype(np.int64)); Uv=torch.from_numpy(np.where(vu<0,NU,vu).astype(np.int64))
if a.nobias: m.b.weight.requires_grad_(False)
opt=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=a.lr,weight_decay=a.wd)
N=len(Lt); bs=2048
sys.path.insert(0,'/tmp/dig'); 
exec(open('/tmp/dig/lat_ev.py').read())
for ep in range(a.ep):
    perm=torch.randperm(N); tot=0
    m.train()
    for i in range(0,N,bs):
        j=perm[i:i+bs]; sc,null=m(Ht[j],Ct[j],Ut[j])
        if a.nonull:
            msk=Lt[j]<10; loss=torch.nn.functional.cross_entropy(sc[msk],Lt[j][msk])
        else:
            loss=torch.nn.functional.cross_entropy(torch.cat([sc,null[:,None]],1),Lt[j])
        opt.zero_grad(); loss.backward(); opt.step(); tot+=loss.item()*len(j)
    m.eval()
    with torch.no_grad():
        u=m.uvec(Hv,Uv).numpy(); Q=m.Q.weight[:NI].numpy() if not a.tied else m.P.weight[:NI].numpy(); b=m.b.weight[:,0].numpy()
        null=(m.c0+(m.wn(torch.from_numpy(u))[:,0] if a.nullu else 0)).numpy()
        if a.user: null=null+m.ub(Uv)[:,0].numpy()
    ev_,ci=V['evix'],V['cix']
    s=(u[ev_]*Q[ci]).sum(1)+b[ci]; dot=(u[ev_]*Q[ci]).sum(1)
    print(f'ep{ep} loss {tot/N:.4f}',end=' '); q(s,'s'); q(dot,'dot',)
np.savez(f'/tmp/dig/tt_{a.tag}.npz',s=s,dot=dot,u=u,Q=Q,b=b,null=null)
