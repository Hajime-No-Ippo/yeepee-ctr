import numpy as np, polars as pl
x=pl.read_parquet('X2.parquet'); R=np.load('ranks.npy')
x=x.with_columns(r3=pl.Series(R[0]),r8=pl.Series(R[1]))
x=x.with_columns(d=pl.col('r3')-pl.col('r8'),
 hlen_b=pl.col('hlen').cut([0,4,9,19],labels=['0','1-4','5-9','10-19','20']),
 user=pl.when(pl.col('ut_n')==0).then(pl.lit('no train events')).otherwise(pl.lit('has train events')),
 item=pl.col('it_n').cut([0,100,1000,10000],labels=['new item','<=100','101-1k','1k-10k','>10k']),
 rec=pl.col('rec_h').cut([6,24,72,168],labels=['<6h','6-24h','1-3d','3-7d','>7d']),
 hourb=pl.col('hour').cut([5,11,17],labels=['0-5','6-11','12-17','18-23']),
 inhist=pl.col('hin')>0, svd32=pl.col('svd32_cos').qcut(5,labels=['q1','q2','q3','q4','q5']))
x=x.with_columns(pl.col('rec').cast(pl.Utf8).fill_null('no history'))
lines=[]
for s in ['hlen_b','user','item','rec','tb','dt','day','hourb','inhist','svd32']:
    t=x.group_by(s).agg(n=pl.len(),p3=pl.col('p3').mean(),v8=pl.col('v8').mean(),r3=pl.col('r3').mean(),r8=pl.col('r8').mean(),absd=pl.col('d').abs().mean(),sp=pl.corr('r3','r8')).sort(s)
    t=t.with_columns(ratio=pl.col('p3')/pl.col('v8'))
    lines.append(f'### {s}\n'+'| seg | rows | p3 mean | v8 mean | p3/v8 | mean rank p3 | mean rank v8 | mean abs rank diff | spearman |\n|---|---|---|---|---|---|---|---|---|\n'+
      '\n'.join(f'| {r[s]} | {r["n"]} | {r["p3"]:.4f} | {r["v8"]:.4f} | {r["ratio"]:.2f} | {r["r3"]:.3f} | {r["r8"]:.3f} | {r["absd"]:.3f} | {r["sp"]:.3f} |' for r in t.iter_rows(named=True)))
open('segments.md','w').write('\n\n'.join(lines)); print(open('segments.md').read())
# biggest disagreements: top 1% |d| profile vs all
top=x.filter(pl.col('d').abs()>=x['d'].abs().quantile(0.99))
for c in ['hlen','ut_n','it_n','rec_h','hin','svd32_cos','day']:
    print(c, 'top1%% disagree mean %.3f vs all %.3f'%(top[c].mean(), x[c].mean()))
print('top1% where p3 higher:', (top['d']>0).mean())
