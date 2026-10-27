import polars as pl, numpy as np, os
os.environ.setdefault('POLARS_MAX_THREADS','4')
D='/workspace/huawei/v3/data/'
ev=pl.read_parquet(D+'ev_eval.parquet'); re=pl.read_parquet(D+'rows_eval.parquet')
et=pl.read_parquet(D+'ev_train.parquet',columns=['eid','u','ts']); rt=pl.read_parquet(D+'rows_train.parquet',columns=['eid','ii','clicked'])
print('train ts', et['ts'].min(), et['ts'].max())
rtt=rt.join(et,on='eid')
gm=rt['clicked'].mean(); print('train ctr',gm)
it=rtt.group_by('ii').agg(it_n=pl.len(),it_c=pl.col('clicked').sum())
# last-7-day item stats
last=et['ts'].max()-7*86400
it7=rtt.filter(pl.col('ts')>=last).group_by('ii').agg(it7_n=pl.len(),it7_c=pl.col('clicked').sum())
ut=rtt.group_by('u').agg(ut_n=pl.len(),ut_c=pl.col('clicked').sum(),ut_ev=pl.col('eid').n_unique(),ut_last=pl.col('ts').max())
del rtt,rt
x=re.join(ev.select('eid','u','ts','tb','dt',pl.col('h').list.len().alias('hlen'),pl.col('h_ts').list.max().alias('hlast')),on='eid',how='left')
# item in own history
hin=re.join(ev.select('eid','h'),on='eid').explode('h').filter(pl.col('h')==pl.col('ii')).group_by('ord').agg(hin=pl.len())
x=x.join(hin,on='ord',how='left').with_columns(pl.col('hin').fill_null(0))
x=x.join(it,on='ii',how='left').join(it7,on='ii',how='left').join(ut,on='u',how='left').with_columns(pl.col(['it_n','it_c','it7_n','it7_c','ut_n','ut_c','ut_ev']).fill_null(0))
x=x.with_columns(it_ctr=(pl.col('it_c')+gm*20)/(pl.col('it_n')+20), it7_ctr=(pl.col('it7_c')+gm*20)/(pl.col('it7_n')+20),
  ut_ctr=(pl.col('ut_c')+gm*20)/(pl.col('ut_n')+20), rec_h=(pl.col('ts')-pl.col('hlast'))/3600, rec_t=(pl.col('ts')-pl.col('ut_last'))/3600,
  day=((pl.col('ts')-1737936000)//86400), hour=((pl.col('ts')%86400)//3600), pos=pl.col('ord')-pl.col('ord').min().over('eid'))
# ---- eval-row aggregates (leak probes) ----
x=x.with_columns(ev_icand=pl.len().over('ii'), ev_uev=pl.col('eid').n_unique().over('u'))
ue=ev.select('eid','u','ts').sort(['u','ts'])
ue=ue.with_columns(ev_ufut=(pl.len().over('u')-pl.int_range(pl.len()).over('u')-1), ev_upast=pl.int_range(pl.len()).over('u'))
x=x.join(ue.select('eid','ev_ufut','ev_upast'),on='eid')
# item in eval histories with h_ts>=eval start (any user)
eh=ev.select('u','h','h_ts').explode(['h','h_ts']).filter(pl.col('h_ts')>=1737936000)
x=x.join(eh.group_by('h').agg(ev_ihist=pl.col('u').n_unique()).rename({'h':'ii'}),on='ii',how='left')
# future click leak: item appears in same user's eval history with h_ts >= this event ts
fu=x.select('ord','u','ii','ts').join(eh.rename({'h':'ii'}),on=['u','ii']).filter(pl.col('h_ts')>=pl.col('ts')).group_by('ord').agg(fut=pl.len(),fut_dt=(pl.col('h_ts')-pl.col('ts')).min())
x=x.join(fu,on='ord',how='left').with_columns(pl.col(['fut','ev_ihist']).fill_null(0))
p=pl.read_parquet('p3.parquet').rename({'predicted_probability':'p3'}); v=pl.read_parquet('v8.parquet').rename({'predicted_probability':'v8'})
x=x.join(p,on='rowid',how='left').join(v,on='rowid',how='left').sort('ord')
x.drop('rowid').write_parquet('X.parquet'); print(x.shape); print(x.null_count().transpose(include_header=True).filter(pl.col('column_0')>0))
print(x.select('fut','fut_dt').describe())
