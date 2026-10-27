import numpy as np, polars as pl
from scipy.stats import norm
x=pl.read_parquet('X33.parquet').join(pl.read_parquet('ranks33.parquet').drop('eid'),on='ord')
M=['rv8','rp3','rp33']
x=x.with_columns([pl.col(m).rank().over('eid').alias('e'+m) for m in M])
cold=x.filter(pl.col('hlen')==0); warm=x.filter(pl.col('hlen')>0)
# item consistency: warm-row item mean of within-event rank -> predicts cold within-event rank?
for m in M:
  im=warm.group_by('ii').agg(im=pl.col('e'+m).mean(),n=pl.len()).filter(pl.col('n')>=20)
  c=cold.join(im,on='ii',how='inner')
  t=c.with_columns(imr=pl.col('im').rank().over('eid')).group_by('eid').agg(r=pl.corr('e'+m,'imr'))
  print(m,'cold: within-event corr with warm item-mean rank',round(t['r'].fill_nan(None).mean(),3),'rows',c.height)
  # cold self-consistency: same item in different cold events
  ic=cold.group_by('ii').agg(n=pl.len(),v=pl.col('e'+m).var()).filter(pl.col('n')>=5)
  print('   cold same-item within-event rank var',round(ic['v'].mean(),2),'(random=8.25)')
print('cold global mean rank pct:',{m:round(cold[m].mean(),3) for m in M}, 'warm:',{m:round(warm[m].mean(),3) for m in M})
print('cold spearman',{f'{a}-{b}':round(cold.select(pl.corr(a,b,method='spearman')).item(),3) for a,b in [('rv8','rp3'),('rv8','rp33'),('rp3','rp33')]})
t=cold.group_by('eid').agg(*[pl.corr('e'+a,'e'+b).alias(f'{a}-{b}') for a,b in [('rv8','rp3'),('rv8','rp33'),('rp3','rp33')]])
print('cold within-event corr', {c:round(t[c].fill_nan(None).mean(),3) for c in t.columns[1:]})
# cold event-level mean rank (between-event): corr with v8's
ev=cold.group_by('eid').agg(*[pl.col(m).mean() for m in M], ut_n=pl.col('ut_n').first())
print('cold event-level corr', ev.select(pl.corr('rv8','rp33'),pl.corr('rp3','rp33').alias('p3'),pl.corr('rv8','rp3').alias('v8p3')).row(0))
