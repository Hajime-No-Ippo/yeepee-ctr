import numpy as np, polars as pl, json, time
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
D='/workspace/huawei/v3/data'; J20=1737331200; DAY=86400; H=3600
TE=pl.read_parquet(f'{D}/ev_train.parquet',columns=['eid','u','ts']); TR=pl.read_parquet(f'{D}/rows_train.parquet',columns=['eid','ii','clicked'])
EE=pl.read_parquet(f'{D}/ev_eval.parquet',columns=['eid','u','ts','h_ts']); ER=pl.read_parquet(f'{D}/rows_eval.parquet',columns=['eid','ii'])
R=TR.join(TE,on='eid').with_columns(((pl.col('ts')-J20)//DAY).alias('day'))
out={}
print('== daily (day0 = Mon 20 Jan 2025)')
dd=R.group_by('day').agg(pl.len().alias('rows'),pl.col('eid').n_unique().alias('events'),pl.col('clicked').mean().alias('row_ctr'),pl.col('ii').n_unique().alias('n_items')).sort('day')
ev=R.group_by('eid','day').agg(pl.col('clicked').max().alias('any')).group_by('day').agg(pl.col('any').mean().alias('event_click_rate')).sort('day')
dd=dd.join(ev,on='day'); print(dd); out['daily']=dd.to_dicts()
# hour-of-day profile per day (share of events by hour bucket 0-6,6-12,12-18,18-24)
hr=TE.with_columns(((pl.col('ts')-J20)//DAY).alias('day'),(((pl.col('ts')-J20)%DAY)//(6*H)).alias('q')).group_by('day','q').len().sort('day','q')
print('events per 6h block', hr.pivot(on='q',index='day',values='len'))
e=EE.with_columns(((pl.col('ts')-J20)//DAY).alias('day'),(((pl.col('ts')-J20)%DAY)//(6*H)).alias('q'))
print('eval events by day', e.group_by('day').len().sort('day').with_columns((pl.col('len')/pl.col('len').sum()).alias('share')))
print('eval Jan27 6h blocks', e.filter(pl.col('day')==7).group_by('q').len().sort('q'))
# item ctr day-to-day drift
ic=R.group_by('day','ii').agg(pl.len().alias('imp'),pl.col('clicked').sum().alias('clk'))
print('== item CTR / popularity drift day d vs d+1 (items with >=50 imp both days)')
res=[]
for d in range(0,6):
    a=ic.filter(pl.col('day')==d); b=ic.filter(pl.col('day')==d+1); j=a.join(b,on='ii',suffix='_n').filter((pl.col('imp')>=50)&(pl.col('imp_n')>=50))
    ca=((j['clk']+1)/(j['imp']+10)).to_numpy(); cb=((j['clk_n']+1)/(j['imp_n']+10)).to_numpy()
    allb=b.join(a,on='ii',how='left',suffix='_p').fill_null(0)
    r=dict(d=d,n_items=len(j),ctr_spearman=round(spearmanr(ca,cb)[0],4),pop_spearman=round(spearmanr(j['imp'],j['imp_n'])[0],4),
           new_item_imp_share=round(float(allb.filter(pl.col('imp_p')==0)['imp'].sum()/allb['imp'].sum()),4))
    # noise ceiling: split day d+1 rows in two random halves -> ctr spearman between halves
    res.append(r); print(r)
out['drift']=res
# split-half ceiling for one day
rng=np.random.default_rng(0); x=R.filter(pl.col('day')==5).with_columns(pl.Series('half',rng.integers(0,2,R.filter(pl.col('day')==5).height)))
h=x.group_by('half','ii').agg(pl.len().alias('imp'),pl.col('clicked').sum().alias('clk'))
j=h.filter(pl.col('half')==0).join(h.filter(pl.col('half')==1),on='ii').filter((pl.col('imp')>=25)&(pl.col('imp_right')>=25))
print('split-half (same day 25, >=25 imp each half) ctr spearman', round(spearmanr((j['clk']+1)/(j['imp']+10),(j['clk_right']+1)/(j['imp_right']+10))[0],4), len(j))
# single-feature AUC of item CTR over different windows, scored on rows of day d (first event per user in [C, C+1d) and all rows)
print('== single-feature row AUC of smoothed item CTR from windows before C, on rows in [C, C+1d)')
P=R.select('ii','ts','clicked').sort('ts'); ts=P['ts'].to_numpy(); ii=P['ii'].to_numpy(); y=P['clicked'].to_numpy().astype(float); NI=50000
def stats(C,w=None,hl=None):
    m=ts<C; m&=(ts>=C-w) if w else m
    wt=np.where(m,1.0,0.0) if hl is None else np.where(ts<C,np.exp(-(C-ts)/(hl/np.log(2))),0.0)
    imp=np.bincount(ii,wt,NI); clk=np.bincount(ii,wt*y,NI); g=clk.sum()/max(imp.sum(),1)
    return (clk+20*g)/(imp+20), imp
aucs=[]
for d in range(2,7):
    C=J20+d*DAY; tgt=(ts>=C)&(ts<C+DAY); r={'cut_day':d}
    for k,(w,hl) in {'all':(None,None),'48h':(48*H,None),'24h':(24*H,None),'12h':(12*H,None),'6h':(6*H,None),'dec6h':(None,6*H),'dec24h':(None,24*H),'dec48h':(None,48*H)}.items():
        c,imp=stats(C,w,hl); r['ctr_'+k]=round(roc_auc_score(y[tgt],c[ii[tgt]]),4); r['pop_'+k]=round(roc_auc_score(y[tgt],imp[ii[tgt]]),4)
    aucs.append(r); print(r)
out['window_auc']=aucs
# item-mix similarity: candidate distribution per train day vs eval candidates (diagnostic only)
print('== candidate item-mix similarity vs Jan-27 eval candidates (cosine of impression shares; JS divergence)')
ev27=ER.join(EE.select('eid','ts'),on='eid').filter(((pl.col('ts')-J20)//DAY)==7)
ve=np.bincount(ev27['ii'].to_numpy(),minlength=NI).astype(float); ve/=ve.sum()
mix=[]
def js(p,q):
    m=(p+q)/2; f=lambda a,b: np.sum(np.where(a>0,a*np.log(a/np.where(b>0,b,1)),0)); return 0.5*f(p,m)+0.5*f(q,m)
for d in range(0,7):
    v=np.bincount(ii[(ts>=J20+d*DAY)&(ts<J20+(d+1)*DAY)],minlength=NI).astype(float); v/=v.sum()
    r=dict(day=d,cos=round(float(v@ve/np.linalg.norm(v)/np.linalg.norm(ve)),4),js=round(float(js(v,ve)),4)); mix.append(r); print(r)
for d in range(0,6):
    a=np.bincount(ii[(ts>=J20+d*DAY)&(ts<J20+(d+1)*DAY)],minlength=NI).astype(float); b=np.bincount(ii[(ts>=J20+(d+1)*DAY)&(ts<J20+(d+2)*DAY)],minlength=NI).astype(float)
    print(' day',d,'vs',d+1,'cos',round(float(a@b/np.linalg.norm(a)/np.linalg.norm(b)),4),'js',round(float(js(a/a.sum(),b/b.sum())),4))
out['mix']=mix
# eval gap structure: time since last history click; and own-history length
g=EE.with_columns((pl.col('ts')-pl.col('h_ts').list.max()).alias('gap'),pl.col('h_ts').list.len().alias('hl'))
print('eval gap quantiles (h)', np.nanpercentile(g['gap'].to_numpy().astype(float),[10,25,50,75,90])/H)
fe=TE.sort('ts').with_columns(((pl.col('ts')-J20)//DAY).alias('day'))
json.dump(out,open('diag.json','w'),default=str,indent=1)
