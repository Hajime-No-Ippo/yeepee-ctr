import numpy as np, polars as pl, lightgbm as lgb
from scipy.stats import norm
from sklearn.metrics import r2_score
x=pl.read_parquet('X33.parquet').join(pl.read_parquet('ranks33.parquet').drop('eid'),on='ord')
x=x.with_columns(zres=pl.Series(np.load('Zres.npy')),stale=(pl.col('ts')-1737936000)/3600,
  cold=(pl.col('hlen')==0).cast(pl.Int8), lit_n=(pl.col('it_n')+1).log(), ltr_h=(pl.col('tr_ihist')+1).log(),
  it_n_rk=pl.col('it_n').rank().over('eid'), it_ctr_rk=pl.col('it_ctr').rank().over('eid'),
  svd_rk=pl.col('svd32_cos').rank().over('eid'), hlog=(pl.col('hlen')+1).log())
for m in ['rv8','rp3','rp33']: x=x.with_columns(pl.Series('z'+m, norm.ppf(np.clip(x[m].to_numpy()-0.5/x.height,1e-7,1-1e-7))))
TR=['hlen','cold','tb','dt','day','hour','stale','pos','lit_n','it_ctr','it7_ctr','ut_ctr','ut_n','ut_ev','rec_h','rec_t','hin','ltr_h','it_ctr_rk','it_n_rk','n1','n6h']
HI=['cocl_sum','cocl_mean','svd8_cos','svd16_cos','svd32_cos','svd64_cos','svd_rk']
eid=x['eid'].to_numpy(); tr=(eid%5)!=0; te=~tr
def fit(cols,y,rounds=300):
    X=x.select(cols).to_numpy().astype(np.float32)
    m=lgb.train(dict(objective='regression',num_leaves=31,max_depth=6,learning_rate=0.1,num_threads=4,verbose=-1,min_data_in_leaf=200),lgb.Dataset(X[tr],y[tr]),rounds)
    return r2_score(y[te],m.predict(X[te])),m
for t in ['zrv8','zrp3','zrp33']:
    y=x[t].to_numpy(); a,_=fit(TR,y); b,m=fit(TR+HI,y)
    print(t,'R2 simple %.3f  +history %.3f'%(a,b))
y=x['zres'].to_numpy(); r,m=fit(TR+HI,y)
imp=sorted(zip(TR+HI,m.feature_importance('gain')),key=lambda t:-t[1]); tot=sum(v for _,v in imp)
print('residual p33|v8,p3: R2 %.3f'%r, ', '.join(f'{k} {v/tot:.2f}' for k,v in imp[:10]))
# p33 ~ v8,p3 + features: incremental info
x=x.with_columns(pl.Series('zv',x['zrv8']),pl.Series('zp',x['zrp3']))
r0,_=fit(['zv','zp'],x['zrp33'].to_numpy()); r1,m1=fit(['zv','zp']+TR+HI,x['zrp33'].to_numpy(),400)
print('p33 ~ v8,p3 tree R2 %.3f; + features %.3f'%(r0,r1))
imp=sorted(zip(['zv','zp']+TR+HI,m1.feature_importance('gain')),key=lambda t:-t[1]); tot=sum(v for _,v in imp)
print('  ', ', '.join(f'{k} {v/tot:.3f}' for k,v in imp[:12]))
# residual by staleness and hlen buckets
x=x.with_columns(sb=pl.col('stale').cut([24,48],labels=['0-24','24-48','48+']))
print(x.group_by('sb').agg(pl.col('zres').mean(),pl.col('zres').std(),pl.corr('zrv8','zrp33'),pl.corr('zrp3','zrp33').alias('c3')).sort('sb'))
