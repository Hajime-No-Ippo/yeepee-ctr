import numpy as np, polars as pl, os
from scipy.stats import norm, spearmanr
x=pl.read_parquet('X33.parquet'); N=x.height
from scipy.stats import rankdata
R={k:rankdata(x[c].to_numpy())/N for k,c in [('v8','v8'),('p3','p3'),('p33','p33r'),('b','b')]}
x=x.with_columns([pl.Series('r'+k,v) for k,v in R.items()])
Z={k:norm.ppf(np.clip(v-0.5/N,1e-7,1-1e-7)) for k,v in R.items()}
print('== global spearman'); ks=list(R)
for i in range(4):
  for j in range(i+1,4): print(ks[i],ks[j],round(np.corrcoef(R[ks[i]],R[ks[j]])[0,1],4))
# within-event rank corr & top1
w=x.select('eid',*[pl.col('r'+k).rank().over('eid').alias('e'+k) for k in ks])
for a,b_ in [('v8','p3'),('v8','p33'),('p3','p33'),('v8','b')]:
  t=w.group_by('eid').agg(c=pl.corr('e'+a,'e'+b_),t1=(pl.col('e'+a).arg_max()==pl.col('e'+b_).arg_max()))
  print(f'within-event {a}-{b_}: mean rank corr {t["c"].mean():.3f}  same top1 {t["t1"].mean():.3f}')
# is p33 a mix of v8 and p3?
A=np.c_[Z['v8'],Z['p3'],np.ones(N)]; be,res,_,_=np.linalg.lstsq(A,Z['p33'],rcond=None)
pred=A@be; r2=1-np.var(Z['p33']-pred)/np.var(Z['p33']); print('p33 normal-score ~ v8+p3:',be.round(3),'R2',round(r2,4))
for wv in [0,0.2,0.3,0.4,0.5,0.6]:
  m=wv*R['v8']+(1-wv)*R['p3']; print(' spearman(p33, %.1f v8 + %.1f p3)'%(wv,1-wv), round(spearmanr(m,R['p33'])[0],4))
np.save('Zres.npy',Z['p33']-pred)
x.select('ord','eid','rv8','rp3','rp33','rb').write_parquet('ranks33.parquet')
