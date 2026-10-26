import polars as pl, numpy as np, math, sys
sys.path.insert(0,'/workspace/huawei/v8')
from feats import load, first_events
from sklearn.metrics import roc_auc_score
D='/workspace/huawei/v8'
TE,TR,EE,ER,NI=load(D)
C=1737763200; W=2*86400
T_ev=first_events(TE,C,W); T_rows=TR.join(T_ev.select('eid'),on='eid')
P_ev=TE.filter(pl.col('ts')<C); P=TR.join(P_ev.select('eid','user'),on='eid')
g=P['clicked'].mean(); K=50
ic=P.group_by('ii').agg(((pl.col('clicked').sum()+K*g)/(pl.len()+K)).alias('ictr'))
uc=P.group_by('user').agg(((pl.col('clicked').sum()+K*g)/(pl.len()+K)).alias('uctr'))
# user_interacted: union of history items of train rows' events + clicked items
inter=pl.concat([P_ev.select('user','h').explode('h').drop_nulls().rename({'h':'ii'}).with_columns(pl.col('ii').cast(pl.Int32)),
                 P.filter(pl.col('clicked')==1).select('user',pl.col('ii').cast(pl.Int32))]).unique()
pop=inter.group_by('ii').agg(pl.len().alias('pop'))
pairs=inter.join(inter,on='user',suffix='_b').filter(pl.col('ii')!=pl.col('ii_b')).group_by('ii','ii_b').agg(pl.len().alias('c'))
pairs=pairs.join(pop,on='ii').join(pop.rename({'ii':'ii_b','pop':'pop_b'}),on='ii_b').with_columns((pl.col('c')/(pl.col('pop')*pl.col('pop_b')).sqrt()).alias('cs')).select('ii','ii_b','cs')
R=T_rows.join(T_ev.select('eid','user','h'),on='eid').with_row_index('r')
PR=R.select('r',pl.col('ii').cast(pl.Int32),'h').explode('h').drop_nulls('h').rename({'h':'ii_b'}).with_columns(pl.col('ii_b').cast(pl.Int32))
cf=PR.join(pairs,on=['ii','ii_b'],how='left').group_by('r').agg(pl.col('cs').fill_null(0).sum().alias('cf'))
R=R.join(ic,on='ii',how='left').join(uc,on='user',how='left').join(cf,on='r',how='left').with_columns(pl.col('ictr').fill_null(g),pl.col('uctr').fill_null(g),pl.col('cf').fill_null(0))
lg=lambda p: np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)))
z=lg(R['ictr'].to_numpy())+lg(R['uctr'].to_numpy())-lg(g)+25*R['cf'].to_numpy()
y=R['clicked'].to_numpy()
print('val events',len(T_ev),'rows',len(R),'ctr',y.mean())
print('starter AUC',roc_auc_score(y,z))
print('item ctr only',roc_auc_score(y,R['ictr'].to_numpy()),'user ctr only',roc_auc_score(y,R['uctr'].to_numpy()),'cf only',roc_auc_score(y,R['cf'].to_numpy()))
print('no cf',roc_auc_score(y,lg(R['ictr'].to_numpy())+lg(R['uctr'].to_numpy())))
