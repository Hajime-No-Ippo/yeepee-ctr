import numpy as np, polars as pl, os
os.chdir('/workspace/huawei/v9')
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score as A
for c in (25,23):
  v=pl.read_parquet(f'feat_v3/val{c}.parquet',columns=['rowid','clicked','us_hlen']).join(pl.read_parquet(f'/workspace/huawei/v8/feat_sqv/all500/val{c}.parquet',columns=['rowid','sq_logit']),on='rowid',how='left',maintain_order='left')
  y=v['clicked'].to_numpy(); hl=v['us_hlen'].to_numpy(); N=len(y)
  st=pl.read_parquet(f'feat_tmB2/val{c}.parquet')['tmR_stale_h'].to_numpy()
  r8=rankdata(.8*(np.load(f'ens/trim_b_c{c}_s0.npy')[:,1]+np.load(f'ens/trim_s_c{c}_s0.npy')[:,2])/2+.2*v['sq_logit'].to_numpy())/N
  b=A(y,r8); res=[]
  for lab,m in [('h1-2',(hl>=1)&(hl<=2)),('h0',hl==0),('stale48+',st>=48),('stale24+',st>=24)]:
    if m.sum()==0: continue
    res.append(lab+' '+' '.join('%+.2f:%+.5f'%(dl,A(y,r8+dl*m)-b) for dl in (-0.04,-0.02,-0.01,0.01,0.02)))
  print(c,'base',round(b,5)); [print('  ',r) for r in res]
