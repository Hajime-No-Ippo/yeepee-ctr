import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[k]='1'
import numpy as np
from scipy.stats import kendalltau, spearmanr
Z=np.load('arrs.npz'); rng=np.random.default_rng(0); idx=rng.choice(529390,200000,replace=False)
def pair(a,b):
    x,y=Z[a][idx],Z[b][idx]
    t=kendalltau(x,y)[0]; return spearmanr(x,y)[0],(1-t)/2
for a in ('U1','U2'):
    for b in ('S1','T1','R1','U2' if a=='U1' else 'U1','EN'):
        s,d=pair(a,b); print(f'{a} vs {b}: spearman {s:.6f}  frac pairs reordered {d:.5f}')
for a,b in (('S1','T1'),('S1','R1'),('T1','R1'),('S1','EN')):
    s,d=pair(a,b); print(f'{a} vs {b}: spearman {s:.6f}  frac pairs reordered {d:.5f}')
