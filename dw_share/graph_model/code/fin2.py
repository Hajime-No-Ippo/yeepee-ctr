import sys,glob,numpy as np,polars as pl; sys.dont_write_bytecode=True
from final import *
sys.path.insert(0,'/workspace/huawei/v21seq')
from evalseq import tologit,yv
G26=np.load(O+'graph_c26.npy') if False else None
for c in (26,27):
    A=score(c); G=zs(A['s_uv'])+zs(A['s_h'])+zs(A['cos_mean']); np.save(O+f'graph_c{c}.npy',G.astype(np.float32)); print(c,G.shape,np.isfinite(G).all())
G=np.load(O+'graph_c26.npy').astype(np.float64); y=yv(26)
fp=sorted(glob.glob('/workspace/huawei/v21seq/ens/ctl6_A_c26_s*.npy')); M=nr(np.mean([tologit(np.load(f)[:,-1]) for f in fp],0)); Gr=nr(G)
print('C26 y rows',len(y),len(G),'graph auc %.5f M auc %.5f rho %.3f'%(AUC(y,G),AUC(y,M),np.corrcoef(Gr,M)[0,1]))
a0=AUC(y,M); print('C26 gain over M (rank): '+' '.join(f'w{w}:{(AUC(y,(1-w)*M+w*Gr)-a0)*1e5:+.0f}' for w in (.05,.1,.125,.15,.2)))
E=np.load(O+'graph_c27.npy'); np.save(O+'graph_eval.npy',E.astype(np.float32))
T=np.load(O+'tab_eval.npz'); np.save(O+'graph_eval_rowid.npy',T['rowid']); print('eval',E.shape,T['rowid'][:3])
