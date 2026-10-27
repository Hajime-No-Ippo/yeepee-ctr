import numpy as np, polars as pl
x=pl.read_parquet('X33s.parquet')
F=['hin','svd32_cos','cocl_sum','it_n','tr_ihist','n1','it_ctr']
for lab,f in [('h1-2',pl.col('hlen').is_between(1,2)),('h3-5',pl.col('hlen').is_between(3,5)),('h11+',pl.col('hlen')>=11)]:
  w=x.filter(f).with_columns([pl.col(c).rank().over('eid').alias('f_'+c) for c in F])
  t=w.group_by('eid').agg([pl.corr(m,'f_'+c).alias(m+'|'+c) for m in ['e8','e3','e33'] for c in F]+[pl.corr('e8','e3').alias('v8~p3'),pl.corr('e8','e33').alias('v8~p33')])
  r={c:round(t[c].fill_nan(None).mean(),3) for c in t.columns[1:]}
  print(lab); [print('  ',m,{c:r[m+'|'+c] for c in F}) for m in ['e8','e3','e33']]; print('   within-event agreement',r['v8~p3'],r['v8~p33'])
# event-level: which event features predict p33's event-mean rank minus v8's?
ev=x.group_by('eid').agg(pl.col('rv8').mean(),pl.col('rp3').mean(),pl.col('rp33').mean(),pl.col('hlen').first(),pl.col('ut_n').first(),pl.col('ut_ctr').first(),pl.col('stale').first(),pl.col('rec_h').first(),hin=pl.col('hin').sum(),svd=pl.col('svd32_cos').max())
for m in ['rv8','rp3','rp33']:
  print('event mean rank',m,'spearman with hlen %.3f ut_ctr %.3f ut_n %.3f stale %.3f rec_h %.3f hin %.3f svdmax %.3f'%tuple(ev.select(pl.corr(m,c,method='spearman')).item() for c in ['hlen','ut_ctr','ut_n','stale','rec_h','hin','svd']))
