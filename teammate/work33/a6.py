import numpy as np, polars as pl
x=pl.read_parquet('X33.parquet').join(pl.read_parquet('ranks33.parquet').drop('eid'),on='ord')
x=x.with_columns(stale=(pl.col('ts')-1737936000)/3600)
x=x.with_columns(sb=pl.col('stale').cut([24,48],labels=['0-24h','24-48h','48h+']),
 hb=pl.col('hlen').cut([0,2,5,10,20],labels=['0','1-2','3-5','6-10','11-20','21+']),
 ib=pl.col('it_n').cut([0,100,1000,10000],labels=['new','<=100','101-1k','1k-10k','>10k']),
 rb_=pl.col('rec_h').cut([24,72,168],labels=['<1d','1-3d','3-7d','>7d']).cast(pl.Utf8).fill_null('no hist'),
 d=pl.col('rb')-pl.col('rv8'), d33=pl.col('rp33')-pl.col('rv8'), d3=pl.col('rp3')-pl.col('rv8'))
x=x.with_columns(dev=pl.col('d').mean().over('eid'), dw=pl.col('d')-pl.col('d').mean().over('eid'),
  e8=pl.col('rv8').rank().over('eid'),eb=pl.col('rb').rank().over('eid'),e33=pl.col('rp33').rank().over('eid'),e3=pl.col('rp3').rank().over('eid'))
TA=(x['d']**2).sum(); TE=(x['dev']**2).sum(); TW=(x['dw']**2).sum()
print('blend-v8 movement: total sq %.1f = event-level %.1f%% + within-event %.1f%%'%(TA,100*TE/TA,100*TW/TA))
x.write_parquet('X33s.parquet')
L=[]
for s in ['sb','hb','rb_','ib','pos','tb']:
  t=x.group_by(s).agg(n=pl.len(),sp33=pl.corr('rv8','rp33',method='spearman'),sp3=pl.corr('rv8','rp3',method='spearman'),
    mv8=pl.col('rv8').mean(),mb=pl.col('rb').mean(),m33=pl.col('rp33').mean(),
    absd=pl.col('d').abs().mean(),sh=(pl.col('d')**2).sum()/TA, shE=(pl.col('dev')**2).sum()/TE, shW=(pl.col('dw')**2).sum()/TW,
    t1=((pl.col('e8')==10)&(pl.col('eb')<10)).sum()/(pl.col('e8')==10).sum()).sort(s)
  t=t.with_columns(rowsh=pl.col('n')/x.height)
  L.append(f'### {s}\n|seg|rows|row share|Spearman v8-p33|Spearman v8-p3|mean rank v8|mean rank blend|mean rank p33|mean abs move|share of move (sq)|share event-level|share within-event|top-1 changed|\n|'+'---|'*13+'\n'+
   '\n'.join(f"|{r[s]}|{r['n']}|{r['rowsh']:.3f}|{r['sp33']:.3f}|{r['sp3']:.3f}|{r['mv8']:.3f}|{r['mb']:.3f}|{r['m33']:.3f}|{r['absd']:.4f}|{r['sh']:.3f}|{r['shE']:.3f}|{r['shW']:.3f}|{r['t1']:.3f}|" for r in t.iter_rows(named=True)))
open('segments33.md','w').write('\n\n'.join(L)); print(open('segments33.md').read())
# stable vs fast signals by staleness: within-event corr
for sb in ['0-24h','24-48h','48h+']:
  w=x.filter((pl.col('sb')==sb)&(pl.col('hlen')>0)).with_columns([pl.col(f).rank().over('eid').alias('f_'+f) for f in ['hin','svd32_cos','cocl_sum','n6h','n1','it_ctr']])
  t=w.group_by('eid').agg([pl.corr(m,'f_'+f).alias(m+'|'+f) for m in ['e8','e3','e33'] for f in ['hin','svd32_cos','cocl_sum','n6h','n1','it_ctr']]+[pl.corr('e8',m).alias('v8~'+m) for m in ['e3','e33']])
  print(sb,{c:round(t[c].fill_nan(None).mean(),3) for c in t.columns[1:]})
# time-since-activity: does p33-v8 shift depend on rec_h/stale (event level)?
ev=x.group_by('eid').agg(pl.col('d33').mean(),pl.col('d3').mean(),pl.col('stale').first(),pl.col('rec_h').first(),pl.col('hlen').first(),pl.col('ut_n').first())
print('event-level corr of mean shift (p33-v8) with stale, rec_h, hlen, ut_n:',[round(ev.select(pl.corr('d33',c,method='spearman')).item(),3) for c in ['stale','rec_h','hlen','ut_n']])
print('event-level corr of mean shift (p3-v8)  with stale, rec_h, hlen, ut_n:',[round(ev.select(pl.corr('d3',c,method='spearman')).item(),3) for c in ['stale','rec_h','hlen','ut_n']])
