from lib import *
A=np.load('A_all.npy')
def dec(x): m=smean(x); return m, x-m
res={}
for nm,x in dict(EN=EN,E0=E0,A=A,P=P).items(): res[nm]=dec(x)
def c(a,b): return np.corrcoef(a,b)[0,1]
print('slate-mean corr: EN-A %.4f EN-E0 %.4f E0-A %.4f EN-P %.4f'%(c(res['EN'][0],res['A'][0]),c(res['EN'][0],res['E0'][0]),c(res['E0'][0],res['A'][0]),c(res['EN'][0],res['P'][0])))
print('within-slate corr: EN-A %.4f EN-E0 %.4f E0-A %.4f EN-P %.4f'%(c(res['EN'][1],res['A'][1]),c(res['EN'][1],res['E0'][1]),c(res['E0'][1],res['A'][1]),c(res['EN'][1],res['P'][1])))
print('var share between-slate: EN %.3f E0 %.3f A %.3f'%tuple(res[k][0].var()/EN.var() if k=='EN' else res[k][0].var()/{'E0':E0,'A':A}[k].var() for k in ['EN','E0','A']))
d=EN-E0; dm,dw=dec(d)
print('d: var between %.5f within %.5f total %.5f ; std d %.4f'%(dm.var(),dw.var(),d.var(),d.std()))
dA=EN-A; print('EN-A: std %.4f between %.5f within %.5f'%(dA.std(),dec(dA)[0].var(),dec(dA)[1].var()))
# buckets
import polars as pl
df=pl.DataFrame(dict(d=d,dm=dm,en=EN,e0=E0,a=A,hl=hl,st=st,dt=meta['day_type'].to_numpy(),tb=meta['time_bucket'].to_numpy()))
df=df.with_columns(pl.col('hl').cut([0,2,6,12,19],labels=['0','1-2','3-6','7-12','13-19','20']).alias('hb'),
 pl.when(pl.col('st').is_null()).then(pl.lit('none')).otherwise(pl.col('st').cut([6,12,24,48,96],labels=['<6h','6-12h','12-24h','24-48h','48-96h','96h+']).cast(pl.Utf8)).alias('sb'))
for k in ['hb','sb','dt','tb']:
    g=df.group_by(k).agg(pl.len().alias('n'),pl.col('en').mean().alias('mEN'),pl.col('e0').mean().alias('mE0'),pl.col('a').mean().alias('mA'),pl.col('d').mean().alias('mean_d'),pl.col('d').std().alias('sd_d'),pl.col('dm').std().alias('sd_dm')).sort(k)
    print(g)
import pickle; df.write_parquet('diagdf.parquet')
