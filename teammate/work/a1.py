import polars as pl, numpy as np, zipfile
from scipy.stats import spearmanr, pearsonr, rankdata
x=pl.read_parquet('X.parquet')
re=pl.read_parquet('/workspace/huawei/v3/data/rows_eval.parquet')
p=pl.read_parquet('p3.parquet'); v=pl.read_parquet('v8.parquet')
print('same order as eval rows:', (p['rowid']==re['rowid']).all(), 'v8 same order:', (v['rowid']==re['rowid']).all())
print('unique rowids', p['rowid'].n_unique(), 'missing', len(set(re['rowid'])-set(p['rowid'])), 'extra', len(set(p['rowid'])-set(re['rowid'])))
a=x['p3'].to_numpy(); b=x['v8'].to_numpy()
for n,q in [('p3',a),('v8',b)]:
    print(n,'mean %.5f std %.5f min %.6f max %.6f'%(q.mean(),q.std(),q.min(),q.max()), 'pct', np.round(np.percentile(q,[1,10,25,50,75,90,99]),5))
    u,c=np.unique(q,return_counts=True); print('  distinct',len(u),'max tie',c.max(),'rows in ties', c[c>1].sum(), 'top vals', u[np.argsort(-c)[:5]], np.sort(c)[::-1][:5])
z=zipfile.ZipFile('../submission_p3_d16.zip'); raw=z.read('submission_p3_d16.csv').decode()
vals=[l.split(',')[1] for l in raw.split('\r\n')[1:6]]; print('raw sample',vals, 'CRLF', '\r\n' in raw[:200]); 
dec=pl.Series([s.split(',')[1].split('.')[-1] for s in raw.strip().split('\r\n')[1:]]).str.len_chars().value_counts(); print(dec)
print(z.infolist()[0].date_time, z.infolist()[0].compress_type, z.infolist()[0].create_system)
# per-event sums
ev=x.group_by('eid').agg(s3=pl.col('p3').sum(),s8=pl.col('v8').sum(),m3=pl.col('p3').max(),n=pl.len(), sd3=pl.col('p3').std())
print(ev.select(pl.col('s3').describe() if False else pl.col('s3').quantile(q).alias(f's3_{q}') for q in [0.01,0.5,0.99]))
print(ev.select([pl.col('s8').quantile(q).alias(f's8_{q}') for q in [0.01,0.5,0.99]]), ev['n'].value_counts())
print('pearson',pearsonr(a,b)[0],'spearman',spearmanr(a,b)[0])
la=np.log(a/(1-a)); lb=np.log(b/(1-b)); print('logit pearson', pearsonr(la,lb)[0])
# within-event spearman
x2=x.with_columns(r3=pl.col('p3').rank().over('eid'), r8=pl.col('v8').rank().over('eid'))
w=x2.group_by('eid').agg(c=pl.corr('r3','r8')); print('mean within-event rank corr', w['c'].mean(), 'top1 agree', x2.group_by('eid').agg(((pl.col('r3')==pl.col('r3').max())&(pl.col('r8')==pl.col('r8').max())).any().alias('t'))['t'].mean())
# between-event: corr of event means
print('event-sum corr', pearsonr(ev['s3'].to_numpy(),ev['s8'].to_numpy())[0])
ra=rankdata(a)/len(a); rb=rankdata(b)/len(b)
qa=np.minimum((ra*10).astype(int),9); qb=np.minimum((rb*10).astype(int),9)
M=np.zeros((10,10),int); np.add.at(M,(qa,qb),1); print('decile table rows=p3 cols=v8 (% of row)\n',np.round(M/M.sum(1,keepdims=True)*100).astype(int))
np.save('ranks.npy',np.vstack([ra,rb]))
