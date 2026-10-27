import numpy as np, polars as pl, lightgbm as lgb
from sklearn.metrics import r2_score
x=pl.read_parquet('X2.parquet')
x=x.with_columns(cold=(pl.col('hlen')==0).cast(pl.Int8), ucold=(pl.col('ut_n')==0).cast(pl.Int8),
  lit_n=(pl.col('it_n')+1).log(), lit7=(pl.col('it7_n')+1).log(), ltr_h=(pl.col('tr_ihist')+1).log(),
  it_ctr_rk=pl.col('it_ctr').rank().over('eid'), it_n_rk=pl.col('it_n').rank().over('eid'),
  lev_icand=(pl.col('ev_icand')+1).log(), lev_ihist=(pl.col('ev_ihist')+1).log(), lev_iday=(pl.col('ev_icand_day')+1).log(), lev_ifut=(pl.col('ev_ifut')+1).log(),
  evc_rk=pl.col('ev_icand').rank().over('eid'), p3l=(pl.col('p3')/(1-pl.col('p3'))).log(), v8l=(pl.col('v8')/(1-pl.col('v8'))).log())
TR=['hlen','cold','ucold','tb','dt','day','hour','pos','lit_n','it_ctr','it7_ctr','lit7','ut_ctr','ut_n','ut_ev','rec_h','rec_t','hin','ltr_h','it_ctr_rk','it_n_rk']
HI=['cocl_sum','cocl_mean','svd8_cos','svd16_cos','svd32_cos','svd64_cos']
EV=['lev_icand','lev_ihist','lev_iday','lev_ifut','evc_rk']
eid=x['eid'].to_numpy(); tr=(eid%5)!=0; te=~tr
def fit(cols,t,rounds=400):
    X=x.select(cols).to_numpy().astype(np.float32); y=x[t].to_numpy()
    m=lgb.train(dict(objective='regression',num_leaves=31,max_depth=6,learning_rate=0.1,num_threads=4,verbose=-1,min_data_in_leaf=200),lgb.Dataset(X[tr],y[tr]),rounds)
    return r2_score(y[te],m.predict(X[te])),m
for t in ['p3l','v8l']:
    r,m=fit(TR+HI,t); re,me=fit(TR+HI+EV,t)
    g=m.feature_importance('gain'); g=g/g.sum(); imp=sorted(zip(TR+HI,g),key=lambda a:-a[1])
    ge=me.feature_importance('gain'); ge=ge/ge.sum()
    print(f'{t}: R2 TR+HIST {r:.3f}; +EV {re:.3f} (delta {re-r:+.4f}); EV gain share {sum(ge[-5:]):.3f}')
    print('   ', ', '.join(f'{a} {b:.3f}' for a,b in imp[:10]))
    for h in HI: 
        rr,_=fit(TR+[h],t,300); print(f'    TR+{h}: {rr:.3f}')
# within-event spearman with history features, by model
xr=x.with_columns([pl.col(c).rank().over('eid').alias('r_'+c) for c in ['p3','v8']+HI+['it_ctr','it_n','hin']])
for c in HI+['it_ctr','it_n','hin']:
    w=xr.filter(pl.col('hlen')>0).group_by('eid').agg(a=pl.corr('r_p3','r_'+c),b=pl.corr('r_v8','r_'+c))
    print(f'within-event rank corr with {c}: p3 {w["a"].mean():.3f}  v8 {w["b"].mean():.3f}')
