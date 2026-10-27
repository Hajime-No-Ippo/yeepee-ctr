import numpy as np, polars as pl
x=pl.read_parquet('X33.parquet').join(pl.read_parquet('ranks33.parquet').drop('eid'),on='ord')
x=x.with_columns(zres=pl.Series(np.load('Zres.npy')))
M=['rv8','rp3','rp33']
cold=x.filter(pl.col('hlen')==0); warm=x.filter(pl.col('hlen')>0)
print('cold rows',cold.height)
for nm,s in [('cold',cold),('warm',warm)]:
  e=s.with_columns([pl.col(c).rank().over('eid').alias('e'+c) for c in M+['it_n','tr_ihist','n1','it_ctr','ev_icand','svd32_cos','cocl_sum','hin']])
  out=[]
  for m in M:
    t=e.group_by('eid').agg([pl.corr('e'+m,'e'+f).alias(f) for f in ['it_n','tr_ihist','n1','it_ctr','ev_icand','svd32_cos','cocl_sum']])
    out.append((m,{f:(lambda v: None if v is None else round(v,3))(t[f].fill_nan(None).drop_nulls().mean()) for f in t.columns if f!='eid'}))
  print(nm,'within-event rank corr with features:'); [print('  ',m,d) for m,d in out]
  pm=e.group_by('pos').agg([pl.col('e'+m).mean().alias(m) for m in M]).sort('pos')
  print(nm,'mean within-event rank by position (1..10):'); print(pm.with_columns(pl.col(M).round(2)).transpose(include_header=True).to_pandas().to_string())
  if nm=='cold':
    for m in M: print('  cold top1 at pos9 share',m, round(e.filter(pl.col('e'+m)==pl.col('e'+m).max().over('eid')).filter(pl.col('pos')==9).height/e['eid'].n_unique(),3))
