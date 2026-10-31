import numpy as np, polars as pl, zipfile, io
from scipy.stats import rankdata, spearmanr
D=pl.read_parquet('../merged.parquet'); n=D.height
z=zipfile.ZipFile('submission_blend_newbest_v21_w10_20261031.csv.zip'); W=pl.read_csv(io.BytesIO(z.read(z.namelist()[0])))['predicted_probability'].to_numpy()
nr=lambda x:(rankdata(x)-1)/(n-1)
O=nr(D['ours'].to_numpy()); E0=nr(D['eb'].to_numpy()); P=nr(D['p33_rank'].to_numpy())
EN=(W-0.1*O)/0.9; EN=np.clip(EN,0,1)
np.save('eb_new_rank.npy',EN)
print('ties in EBnew rank (unique/n)',len(np.unique(np.round(EN,9))),n,'old EB unique',len(np.unique(E0)),'ours unique',len(np.unique(O)))
D=D.with_columns(pl.Series('en',EN),pl.Series('e0',E0),pl.Series('o',O),pl.Series('p',P))
print('slate sizes',D.group_by('event_id').len()['len'].value_counts().sort('len'))
print('p33 null',D['p33_rank'].null_count())
sp=lambda a,b:spearmanr(a,b)[0]
print('Spearman EBnew-EBold',sp(EN,E0),'EBnew-ours',sp(EN,O),'EBold-ours',sp(E0,O),'EBnew-p33',sp(EN,P),'EBold-p33',sp(E0,P),'ours-p33',sp(O,P))
# within-slate
def top1(a,b):
    g=D.with_columns(pl.Series('a',a),pl.Series('b',b))
    t=g.group_by('event_id').agg(pl.col('a').arg_max().alias('ia'),pl.col('b').arg_max().alias('ib'),pl.len().alias('n'))
    return (t['ia']==t['ib']).mean()
def top3(a,b):
    g=D.with_columns(pl.Series('a',a),pl.Series('b',b))
    g=g.with_columns(pl.col('a').rank('ordinal',descending=True).over('event_id').alias('ra'),pl.col('b').rank('ordinal',descending=True).over('event_id').alias('rb'))
    return g.filter(pl.col('ra')<=3).filter(pl.col('rb')<=3).group_by('event_id').len()['len'].mean()
for nm,a,b in (('EBnew vs EBold',EN,E0),('EBnew vs ours',EN,O),('EBold vs ours',E0,O),('EBnew vs p33',EN,P),('EBold vs p33',E0,P)):
    print(nm,'slate top1 agree',round(top1(a,b),4),'avg overlap of top3 sets',round(top3(a,b),3))
# within-slate demeaned pearson
def wp(a,b):
    g=D.with_columns(pl.Series('a',a),pl.Series('b',b)).with_columns((pl.col('a')-pl.col('a').mean().over('event_id')).alias('a'),(pl.col('b')-pl.col('b').mean().over('event_id')).alias('b'))
    return np.corrcoef(g['a'],g['b'])[0,1]
def sm(a,b):
    g=D.with_columns(pl.Series('a',a),pl.Series('b',b)).group_by('event_id').agg(pl.col('a').mean(),pl.col('b').mean())
    return sp(g['a'],g['b'])
for nm,a,b in (('EBnew vs EBold',EN,E0),('EBnew vs ours',EN,O),('EBold vs ours',E0,O)):
    print(nm,'within-slate demeaned pearson',round(wp(a,b),4),'slate-mean spearman',round(sm(a,b),4))
# change
d=EN-E0
print('change: mean',d.mean(),'std',d.std(),'mean|d|',np.abs(d).mean(),'p99|d|',np.percentile(np.abs(d),99))
D=D.with_columns(pl.Series('d',d))
# within-slate change
g=D.with_columns((pl.col('en')-pl.col('en').mean().over('event_id')).alias('en_c'),(pl.col('e0')-pl.col('e0').mean().over('event_id')).alias('e0_c'),(pl.col('o')-pl.col('o').mean().over('event_id')).alias('o_c'),(pl.col('p')-pl.col('p').mean().over('event_id')).alias('p_c'))
g=g.with_columns((pl.col('en_c')-pl.col('e0_c')).alias('dc'))
for c in ('o','p','e0','en','o_c','p_c','e0_c'):
    print('corr(d,%s) %.4f ; corr(within-slate d,%s) %.4f'%(c,np.corrcoef(g['d'],g[c])[0,1],c,np.corrcoef(g['dc'],g[c])[0,1]))
print('corr(d, ours - eb0 gap)',np.corrcoef(g['d'],g['o']-g['e0'])[0,1],' corr(dc, o_c-e0_c)',np.corrcoef(g['dc'],g['o_c']-g['e0_c'])[0,1])
# per-slate: movement of slate mean, and did top1 change
S=g.group_by('event_id').agg(pl.col('d').mean().alias('dm'),pl.col('dc').abs().mean().alias('adc'),pl.col('en').mean().alias('enm'),pl.col('e0').mean().alias('e0m'),pl.col('o').mean().alias('om'),pl.len().alias('n'))
print('slate-mean d: std',S['dm'].std(),'; within-slate |dc| mean',S['adc'].mean())
print('var decomposition of d: slate-mean part',np.var(g['d']-g['dc']) / np.var(g['d']) if False else (np.var(g['d']-g['dc'])/np.var(g['d'])))
# by slate mean old EB tercile
S=S.with_columns(pl.col('e0m').qcut(5,labels=list('12345')).alias('q'))
print(S.group_by('q').agg(pl.col('dm').mean(),pl.col('adc').mean(),pl.len()).sort('q'))
# rows with biggest change: quantile of e0 bin
g=g.with_columns(pl.col('e0').qcut(10,labels=[str(i) for i in range(10)]).alias('b'))
print(g.group_by('b').agg(pl.col('d').mean().alias('mean_d'),pl.col('d').abs().mean().alias('mean_abs_d'),pl.col('d').std().alias('sd')).sort('b'))
g.select(['event_id','rowid','e0','en','o','p','d']).write_parquet('cmp.parquet')
