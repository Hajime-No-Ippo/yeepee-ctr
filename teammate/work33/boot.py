import numpy as np, polars as pl, os
os.chdir('/workspace/huawei/v9')
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score as A
c=25
v=pl.read_parquet(f'feat_v3/val{c}.parquet',columns=['rowid','eid','clicked']).join(pl.read_parquet(f'/workspace/huawei/v8/feat_sqv/all500/val{c}.parquet',columns=['rowid','sq_logit']),on='rowid',how='left',maintain_order='left')
y=v['clicked'].to_numpy(); e=v['eid'].to_numpy(); N=len(y)
z=lambda t: .8*(np.load(f'ens/{t}_b_c{c}_s0.npy')[:,1]+np.load(f'ens/{t}_s_c{c}_s0.npy')[:,2])/2+.2*v['sq_logit'].to_numpy()
r8=rankdata(z('trim'))/N; rp=rankdata(np.load(f'ens/p3like_b_c{c}_s0.npy')[:,1])/N; bl=.7*r8+.3*rp
ue,inv=np.unique(e,return_inverse=True); rng=np.random.default_rng(0); D=[]
for frac in (1.0,0.3):
  D=[]
  for _ in range(60):
    w=rng.poisson(1.0,len(ue)).astype(float)*(rng.random(len(ue))<frac); ww=w[inv]; m=ww>0
    D.append(A(y[m],bl[m],sample_weight=ww[m])-A(y[m],r8[m],sample_weight=ww[m]))
  print('events frac',frac,'paired d_auc mean %.5f sd %.5f'%(np.mean(D),np.std(D)))
