import os
os.environ.update(OMP_NUM_THREADS='2')
import numpy as np, sys
sys.path.insert(0,'.'); from diag1 import *
meta=load_meta(25); y=meta['clicked'].to_numpy(); z=blend('B',25)[1]; E=len(y)//10; Z=z.reshape(E,10); has=y.reshape(E,10).max(1)
from scipy.special import logsumexp
# residual slate-level info beyond the model: add a*(has - 0.5)*sigma-scaled + noise; measure slate AUC of (lse + added) and global AUC
rng=np.random.RandomState(0); base=AUC(y,z); lse=logsumexp(Z,1); bs=AUC(has,lse)
print('base global',round(base,5),'slate AUC(lse)',round(bs,5))
from scipy.special import expit, logit, softmax
p=expit(Z); prior=np.clip(1-np.prod(1-p,1),1e-4,1-1e-4); share=softmax(Z,1)
pr=(prior[:,None]*share).ravel(); print('row prob = prior_has*share:',round(AUC(y,pr),5))
for d in (0.1,0.2,0.3,0.5,0.8,1.2):
    sg=d*has+rng.randn(E); post=expit(logit(prior)+d*sg-d*d/2)
    sl=AUC(has,post); g=AUC(y,(post[:,None]*share).ravel())
    print(f'evidence d={d}: slate AUC of posterior {sl:.4f} (+{sl-AUC(has,prior):.4f})  global={g:.5f} (+{g-AUC(y,pr):.5f})')
