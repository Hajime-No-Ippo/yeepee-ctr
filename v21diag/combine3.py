import os; os.environ['OMP_NUM_THREADS']='2'
import sys, json, numpy as np
sys.path.insert(0,'.'); from diag1 import *
from scipy.special import logsumexp, expit
def get(cut,v,which):
    meta=load_meta(cut); y=meta['clicked'].to_numpy()
    z=blend(v,cut)[1] if which=='blend' else np.load(f'/workspace/huawei/v19/ens/nm{v}s_c{cut}_s0.npy')[:,0]
    return y,z.reshape(-1,10)
def comb(Z,ps,a,c): lse=logsumexp(Z,1,keepdims=True); return (Z - c*lse + a*ps[:,None]).ravel()
if __name__=='__main__':
    res={}
    for v in sys.argv[1:]:
        ps={c:np.load(f'ens/pslate_c{c}_{v}.npy') for c in (23,25)}
        for ri in range(3):
            y3,Z3=get(23,v,'dwbig'); has3=y3.reshape(-1,10).max(1); P3=ps[23][:,ri]; P3=(P3-P3.mean())  # raw logit centred
            base3=AUC(y3,Z3.ravel()); best=(0,0,0)
            for c in (0,0.25,0.5,0.75,1.0):
                for a in (0,0.1,0.2,0.3,0.5,0.75,1.0,1.5):
                    g=AUC(y3,comb(Z3,P3,a,c).ravel())-base3
                    if g>best[0]: best=(g,a,c)
            g3,a,c=best; out=dict(v=v,round_idx=ri,C23_best_delta=round(g3,5),a=a,c=c,slateAUC23=round(AUC(has3,ps[23][:,ri]),4))
            for which in ('dwbig','blend'):
                y,Z=get(25,v,which); P=ps[25][:,ri]; P=P-P.mean(); b=AUC(y,Z.ravel())
                out[f'C25_{which}_base']=round(b,5); out[f'C25_{which}_delta@C23tuned']=round(AUC(y,comb(Z,P,a,c).ravel())-b,5)
                out[f'C25_{which}_oracle_grid_delta']=round(max(AUC(y,comb(Z,P,aa,cc).ravel())-b for cc in (0,.5,1) for aa in (.1,.3,.5,1)),5)
            out['slateAUC25']=round(AUC(y.reshape(-1,10).max(1),ps[25][:,ri]),4); print(json.dumps(out),flush=True)
