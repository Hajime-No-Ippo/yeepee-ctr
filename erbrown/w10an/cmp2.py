import numpy as np, polars as pl
from scipy.stats import spearmanr
g=pl.read_parquet('cmp.parquet')
ev=pl.read_csv('../../data/evaluation.csv',columns=['rowid','user_id','item_id','timestamp','time_bucket','day_type','history_item_ids','history_timestamps'])
ev=ev.with_columns(pl.col('history_item_ids').str.count_matches('itm_').alias('hlen'),
  pl.col('history_timestamps').str.extract(r'(\d+)\D*$',1).cast(pl.Int64).alias('lastts'))
ev=ev.with_columns(((pl.col('timestamp')-pl.col('lastts'))/86400).alias('stale_days'))
ev=ev.with_columns(pl.len().over('item_id').alias('ifreq'),pl.len().over('user_id').alias('ufreq'))
g=g.join(ev.select(['rowid','timestamp','time_bucket','day_type','hlen','stale_days','ifreq','ufreq']),on='rowid')
g=g.with_columns(*(( pl.col(c)-pl.col(c).mean().over('event_id')).alias(c+'_c') for c in ['e0','en','o','p','d']))
print(g['hlen'].describe(), g['stale_days'].describe())
# regression of d_c on e0_c,o_c,p_c (within slate)
X=np.c_[g['e0_c'],g['o_c'],g['p_c']]; y=g['d_c'].to_numpy()
b=np.linalg.lstsq(X,y,rcond=None)[0]; print('d_c ~ e0_c,o_c,p_c coefs',b,'R2',1-((y-X@b)**2).sum()/(y**2).sum())
# EN_c ~ e0_c,o_c,p_c vs old
for nm in ('en','e0'):
    y=g[nm+'_c'].to_numpy(); X=np.c_[g['o_c'],g['p_c']]; bb=np.linalg.lstsq(X,y,rcond=None)[0]; print(nm,'~ o_c,p_c coefs',bb)
g=g.with_columns(pl.col('d_c').abs().alias('ad'))
for c in ['hlen','stale_days','ifreq','ufreq','timestamp','day_type','time_bucket']:
    print(c,'spearman with |d_c|',round(spearmanr(g[c],g['ad'])[0],4),'| with d (row-level)',round(spearmanr(g[c],g['d'])[0],4), '| with en-e0 slate-mean',)
# slate level: row's slate-level user stale
S=g.group_by('event_id').agg(pl.col('d').mean().alias('dm'),pl.col('ad').mean().alias('ad'),pl.col('stale_days').first().alias('st'),pl.col('hlen').first().alias('hl'),pl.col('e0').mean().alias('e0m'),pl.col('o').mean().alias('om'),pl.col('timestamp').first().alias('ts'))
for c in ['st','hl','ts','e0m','om']:
    print('slate',c,'spearman with mean d',round(spearmanr(S[c],S['dm'])[0],4),'with within |d_c|',round(spearmanr(S[c],S['ad'])[0],4))
# bins by stale
S=S.with_columns(pl.col('st').qcut(5,allow_duplicates=True).alias('q'))
print(S.group_by('q').agg(pl.col('dm').mean(),pl.col('ad').mean(),pl.len()).sort('q'))
S=S.with_columns(pl.col('hl').qcut(5,allow_duplicates=True).alias('q2'))
print(S.group_by('q2').agg(pl.col('dm').mean(),pl.col('ad').mean(),pl.len()).sort('q2'))
S=S.with_columns(pl.col('ts').qcut(5,allow_duplicates=True).alias('q3'))
print(S.group_by('q3').agg(pl.col('dm').mean(),pl.col('ad').mean(),pl.len()).sort('q3'))
# top-1 changed slates
t=g.group_by('event_id').agg(pl.col('e0').arg_max().alias('a'),pl.col('en').arg_max().alias('b'),pl.col('o').arg_max().alias('c'))
t=t.with_columns((pl.col('a')!=pl.col('b')).alias('chg'),(pl.col('b')==pl.col('c')).alias('en_eq_o'),(pl.col('a')==pl.col('c')).alias('e0_eq_o'))
print('top1 changed',t['chg'].mean(),'of changed slates: new top == ours top',t.filter(pl.col('chg'))['en_eq_o'].mean(),'old top == ours top',t.filter(pl.col('chg'))['e0_eq_o'].mean())
# 3-way top1 agree
print('top1 all three agree',((t['a']==t['b'])&(t['b']==t['c'])).mean())
# old v new: how often ours top1 is correct? unknown
