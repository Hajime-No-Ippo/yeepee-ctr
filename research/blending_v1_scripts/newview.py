"""New-view member screen. usage: python newview.py CUT  (CUT in 25,23,26). Train-only: uses lib20.load(cut) = f-tables before the cutoff + val{cut}. NEVER touches eval.
Members: leaf_lr (GBDT leaf one-hot -> LR), leaf_fm (leaf fields -> torch FM), gdrop (GBDT w/o top-30 gain feats, feature-bagged view)."""
import os,sys,json,time
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='2'
os.environ['V20_THREADS']='2'
sys.path.insert(0,'/workspace/huawei/v20')
import numpy as np, lightgbm as lgb, scipy.sparse as sp, torch
import lib20
from sklearn.metrics import roc_auc_score as AUC
from sklearn.linear_model import LogisticRegression
torch.set_num_threads(2); torch.manual_seed(0); np.random.seed(0)
cut=int(sys.argv[1]); H='/workspace/huawei'; OUT=f'{H}/research/blending_v1_scripts/out'
t0=time.time(); lg=lambda *a:print(f'[{time.time()-t0:6.0f}s]',*a,flush=True)
gain=json.load(open(f'{H}/v20/ens/ref484_c25_s0_gain.json')); rank=sorted(gain,key=gain.get,reverse=True)
Xt,yt,Xv,yv,fl=lib20.load(cut); lg('loaded',Xt.shape,Xv.shape)
rid=lib20._C['rowid'].copy(); z=np.load(f'{H}/cb_blend/oof_c{cut}.npz',allow_pickle=True); assert (z['rowid']==rid).all() and (z['y']==yv).all(); lg('rowid aligned')
ix=lambda names:[fl.index(c) for c in names]
K=250; keep=rank[:K]; kk=ix(keep); Xt=np.ascontiguousarray(Xt[:,kk]); Xv=np.ascontiguousarray(Xv[:,kk]); fl=keep; lib20._C.clear(); import gc; gc.collect(); lg('sliced',Xt.shape)
# user split for GBDT-vs-LR halves: use row parity of a hash of position is not user-based; load() does not return user ids -> use contiguous-block halves of the table order is biased; use random 50/50 by rows (leaf model sees other rows than LR)
rs=np.random.RandomState(0); half=rs.rand(len(yt))<0.5
res={}
def save(name,s): np.save(f'{OUT}/{name}_c{cut}.npy',s.astype(np.float32)); res[name]=round(AUC(yv,s),5); lg(name,'standalone AUC',res[name])
# ---- (a) GBDT leaves
top=[fl.index(c) for c in rank[:200]]
P=dict(objective='binary',learning_rate=0.1,num_leaves=31,min_data_in_leaf=200,feature_fraction=0.5,bagging_fraction=0.8,bagging_freq=1,lambda_l2=30,verbose=-1,num_threads=2,seed=0,max_bin=63)
NT=100
g=lgb.train(P,lgb.Dataset(Xt[half][:,top],yt[half]),NT); lg('leaf gbdt trained; its val AUC',round(AUC(yv,g.predict(Xv[:,top],raw_score=True)),5))
res['leafgbdt_val']=round(AUC(yv,g.predict(Xv[:,top],raw_score=True)),5)
def leaves(X): return np.concatenate([g.predict(X[i:i+100000][:,top],pred_leaf=True).astype(np.int16) for i in range(0,len(X),100000)])
Lb=leaves(Xt[~half]); Lv=leaves(Xv); yb=yt[~half]; lg('leaves done')
NL=P['num_leaves']
def onehot(L): 
    n=len(L); col=(L.astype(np.int64)+np.arange(NT)[None,:]*NL).ravel(); return sp.csr_matrix((np.ones(len(col),np.float32),(np.repeat(np.arange(n),NT),col)),shape=(n,NT*NL))
Ob,Ov=onehot(Lb),onehot(Lv)
for C in (0.03,0.3):
    m=LogisticRegression(C=C,solver='lbfgs',max_iter=150).fit(Ob,yb); s=m.decision_function(Ov); lg('C',C,round(AUC(yv,s),5)); save(f'leaf_lr_C{C}',s)
# torch FM over leaf fields
class FM(torch.nn.Module):
    def __init__(s,nf,k=8):
        super().__init__(); s.V=torch.nn.Embedding(nf,k); s.w=torch.nn.Embedding(nf,1); torch.nn.init.normal_(s.V.weight,0,0.01); torch.nn.init.zeros_(s.w.weight); s.b=torch.nn.Parameter(torch.tensor(-2.7))
    def forward(s,idx):
        v=s.V(idx); sq=(v.sum(1)**2-(v**2).sum(1)).sum(1)*0.5; return s.b+s.w(idx).sum((1,2))+sq
idxb=torch.from_numpy((Lb.astype(np.int64)+np.arange(NT)[None,:]*NL)); idxv=torch.from_numpy((Lv.astype(np.int64)+np.arange(NT)[None,:]*NL)); ybt=torch.from_numpy(yb.astype(np.float32))
fm=FM(NT*NL); opt=torch.optim.Adam(fm.parameters(),lr=3e-3,weight_decay=0)
best=None
for ep in range(4):
    perm=torch.randperm(len(ybt)); 
    for i in range(0,len(perm),4096):
        b=perm[i:i+4096]; loss=torch.nn.functional.binary_cross_entropy_with_logits(fm(idxb[b]),ybt[b])+1e-6*(fm.V(idxb[b])**2).sum()/len(b)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): s=torch.cat([fm(idxv[i:i+65536]) for i in range(0,len(idxv),65536)]).numpy()
    lg('FM ep',ep,round(AUC(yv,s),5)); np.save(f'{OUT}/leaf_fm_ep{ep}_c{cut}.npy',s.astype(np.float32)); res[f'leaf_fm_ep{ep}']=round(AUC(yv,s),5)
del Ob,Ov,idxb,Lb; 
# ---- (c) feature-bagged view: drop top-30 gain features, extra_trees
drop=set(rank[:30]); use=[i for i,c in enumerate(fl) if c not in drop]
P2=dict(objective='binary',learning_rate=0.05,num_leaves=63,min_data_in_leaf=300,feature_fraction=0.3,bagging_fraction=0.7,bagging_freq=1,lambda_l2=50,extra_trees=True,verbose=-1,num_threads=2,seed=1,max_bin=63)
g2=lgb.train(P2,lgb.Dataset(Xt[:,use],yt),400); s=g2.predict(Xv[:,use],raw_score=True); save('gdrop',s)
json.dump(res,open(f'{OUT}/newview_res_c{cut}.json','w'),indent=1); lg('done')
