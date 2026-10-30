import numpy as np, polars as pl, zipfile, io
from scipy.stats import rankdata, spearmanr, pearsonr
H='/workspace/huawei'
ev=pl.read_csv(f'{H}/data/evaluation.csv',columns=['rowid','event_id','user_id','item_id'])
def rd(zp):
    z=zipfile.ZipFile(zp); return pl.read_csv(io.BytesIO(z.read(z.namelist()[0])))
eb=rd('submission_best.csv.zip'); ours=rd(f'{H}/v21diag/out/submission_v21list_B_TRANSDUCTIVE.csv.zip'); pr=rd(f'{H}/v21diag/out/probe_v21list_B_TRANSDUCTIVE_p33_w80.csv.zip')
p33=pl.read_parquet(f'{H}/teammate/work33/p33_recovered_ranks.parquet').select(['rowid','p33_rank'])
print('rows',eb.shape,ours.shape,pr.shape,'cols',eb.columns,ours.columns)
print('EB same ids set',set(eb['rowid'])==set(ev['rowid']),'dup',eb['rowid'].n_unique(),'same order as eval',eb['rowid'].to_list()==ev['rowid'].to_list())
s=eb['predicted_probability'].to_numpy(); print('EB min/max/mean/median',s.min(),s.max(),s.mean(),np.median(s),'nan',np.isnan(s).sum(),'unique',len(np.unique(s)),'dtype',eb['predicted_probability'].dtype)
for n,d in(('ours',ours),('probe',pr)):
    x=d['predicted_probability'].to_numpy();print(n,'min/max/mean',x.min(),x.max(),x.mean(),'unique',len(np.unique(x)))
D=ev.join(eb.rename({'predicted_probability':'eb'}),on='rowid').join(ours.rename({'predicted_probability':'ours'}),on='rowid').join(pr.rename({'predicted_probability':'pr'}),on='rowid').join(p33,on='rowid',how='left')
print('p33 null',D['p33_rank'].null_count(),'rows',D.height)
D.write_parquet('merged.parquet')
cols=['eb','ours','pr','p33_rank']
R={c:rankdata(D[c].to_numpy()) for c in cols}
for c in cols[1:]:
    print(f'EB vs {c}: spearman {spearmanr(D["eb"],D[c])[0]:.5f} pearson(raw) {pearsonr(D["eb"],D[c])[0]:.5f} pearson(rank) {pearsonr(R["eb"],R[c])[0]:.5f}')
print('ours vs p33 spearman',spearmanr(D['ours'],D['p33_rank'])[0],'pr vs p33',spearmanr(D['pr'],D['p33_rank'])[0])
# slates
print('events',D['event_id'].n_unique(),'rows per event',D.group_by('event_id').len()['len'].value_counts().sort('len').head())
def within(c):
    g=D.with_columns(pl.col(c).rank().over('event_id').alias('a'),pl.col('eb').rank().over('event_id').alias('b'))
    a=g['a'].to_numpy();b=g['b'].to_numpy()
    # per-event spearman averaged: pearson of within-ranks by event
    h=g.group_by('event_id').agg(((pl.col('a')-pl.col('a').mean())*(pl.col('b')-pl.col('b').mean())).sum().alias('cv'),pl.col('a').std().alias('sa'),pl.col('b').std().alias('sb'),pl.len().alias('n'))
    h=h.with_columns((pl.col('cv')/((pl.col('n')-1)*pl.col('sa')*pl.col('sb'))).alias('r')).drop_nulls('r')
    top=g.filter(pl.col('a')==pl.col('a').max().over('event_id')).group_by('event_id').len() # skip
    # top1 agreement
    t1=lambda col:(D.sort(col,descending=True).group_by('event_id',maintain_order=True).first().select('event_id','rowid').rename({'rowid':'r_'+col}))
    m=t1('eb').join(t1(c),on='event_id'); ag=(m['r_eb']==m['r_'+c]).mean()
    return h['r'].mean(),h['r'].median(),ag
for c in cols[1:]: print('within-slate mean/median spearman, top1 agree',c,within(c))
S=D.group_by('event_id').agg(*[pl.col(c).mean().alias(c) for c in cols])
for c in cols[1:]: print('slate-mean spearman EB vs',c,spearmanr(S['eb'],S[c])[0])
# within-slate demeaned correlation
# item freq
cnt=D.group_by('item_id').len().rename({'len':'ic'}); D=D.join(cnt,on='item_id')
ic=D['ic'].to_numpy()
print('items',cnt.height)
for c in cols: print('spearman vs eval item impressions',c,spearmanr(D[c],ic)[0])
# user-level
uc=D.group_by('user_id').len().rename({'len':'uc'}); D=D.join(uc,on='user_id'); print('spearman vs user rows', {c:round(spearmanr(D[c],D['uc'])[0],4) for c in cols})
