import numpy as np, polars as pl, lightgbm as lgb
from sklearn.metrics import r2_score
x=pl.read_parquet('X.parquet')
x=x.with_columns(cold=(pl.col('hlen')==0).cast(pl.Int8), ucold=(pl.col('ut_n')==0).cast(pl.Int8),
  lit_n=(pl.col('it_n')+1).log(), lit7=(pl.col('it7_n')+1).log(), ltr_h=(pl.col('tr_ihist')+1).log(),
  it_ctr_rk=pl.col('it_ctr').rank().over('eid'), it_n_rk=pl.col('it_n').rank().over('eid'),
  lev_icand=(pl.col('ev_icand')+1).log(), lev_ihist=(pl.col('ev_ihist')+1).log(), lev_iday=(pl.col('ev_icand_day')+1).log(), lev_ifut=(pl.col('ev_ifut')+1).log(),
  evc_rk=pl.col('ev_icand').rank().over('eid'))
TR=['hlen','cold','ucold','tb','dt','day','hour','pos','lit_n','it_ctr','it7_ctr','lit7','ut_ctr','ut_n','ut_ev','rec_h','rec_t','hin','ltr_h','it_ctr_rk','it_n_rk']
EV=['lev_icand','lev_ihist','lev_iday','lev_ifut','evc_rk']
lg=lambda p: np.log(p/(1-p))
tgt={'p3':lg(x['p3'].to_numpy()),'v8':lg(x['v8'].to_numpy())}
eid=x['eid'].to_numpy(); tr=(eid%5)!=0; te=~tr   # split by event
def fit(cols,y,depth=6,rounds=300):
    X=x.select(cols).to_numpy().astype(np.float32)
    m=lgb.train(dict(objective='regression',num_leaves=2**depth-1 if depth<6 else 31,max_depth=depth,learning_rate=0.1,num_threads=4,verbose=-1,min_data_in_leaf=200),lgb.Dataset(X[tr],y[tr]),rounds)
    return r2_score(y[te],m.predict(X[te])), m
out={}
for n,y in tgt.items():
    r1,_=fit(['it_ctr'],y,2,50); r_lin=None
    from sklearn.linear_model import LinearRegression
    X=x.select(TR).fill_null(-1).to_numpy(); r_lin=r2_score(y[te],LinearRegression().fit(X[tr],y[tr]).predict(X[te]))
    r3,_=fit(TR,y,3,200); r6,m6=fit(TR,y,6,400); r6e,m6e=fit(TR+EV,y,6,400)
    imp=sorted(zip(TR,m6.feature_importance('gain')),key=lambda t:-t[1]); tot=sum(v for _,v in imp)
    impe=sorted(zip(TR+EV,m6e.feature_importance('gain')),key=lambda t:-t[1]); tote=sum(v for _,v in impe)
    print(f'== {n}: R2 it_ctr-only {r1:.3f} | linear {r_lin:.3f} | tree d3 {r3:.3f} | tree d6 {r6:.3f} | +eval-aggs {r6e:.3f} (delta {r6e-r6:+.4f})')
    print('  gain share:', ', '.join(f'{k} {v/tot:.3f}' for k,v in imp[:12]))
    print('  +EV gain share:', ', '.join(f'{k} {v/tote:.3f}' for k,v in impe if k in EV))
# p3 explained by v8 + simple features; residual vs eval aggs
y=tgt['p3']; v=tgt['v8']
x=x.with_columns(v8l=pl.Series(v))
r,m=fit(TR+['v8l'],y,6,400); re,me=fit(TR+['v8l']+EV,y,6,400)
print(f'p3 ~ v8+TR: R2 {r:.3f}; +EV {re:.3f} (delta {re-r:+.4f})')
r,_=fit(['v8l'],y,6,200); print('p3 ~ v8 only', round(r,3))
yv=tgt['v8']; x=x.with_columns(p3l=pl.Series(y))
r,_=fit(TR+['p3l'],yv,6,400); re,_=fit(TR+['p3l']+EV,yv,6,400); print(f'control v8 ~ p3+TR: {r:.3f}; +EV {re:.3f} (delta {re-r:+.4f})')
