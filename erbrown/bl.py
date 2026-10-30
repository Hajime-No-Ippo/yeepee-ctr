import numpy as np, polars as pl, zipfile, hashlib, os
from scipy.stats import rankdata, spearmanr
D=pl.read_parquet('merged.parquet'); n=D.height
nr=lambda x:(rankdata(x)-1)/(n-1)
r={c:nr(D[c].to_numpy()) for c in ['eb','ours','pr','p33_rank']}
# slate mean of ranks
for c in ['ours','pr','p33_rank']:
    S=D.with_columns(pl.Series('a',r['eb']),pl.Series('b',r[c])).group_by('event_id').agg(pl.col('a').mean(),pl.col('b').mean())
    print('slate-mean-of-rank spearman EB vs',c,round(spearmanr(S['a'],S['b'])[0],5))
# within-slate demeaned rank pearson (global)
for c in ['ours','p33_rank']:
    G=D.with_columns(pl.Series('a',r['eb']),pl.Series('b',r[c])).with_columns((pl.col('a')-pl.col('a').mean().over('event_id')).alias('a'),(pl.col('b')-pl.col('b').mean().over('event_id')).alias('b'))
    print('pooled within-slate demeaned-rank pearson EB vs',c,round(np.corrcoef(G['a'],G['b'])[0,1],5))
def zipit(csv):
    zp=csv+'.zip'; z=zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED); z.write(csv,arcname=os.path.basename(csv)); z.close()
    sh=hashlib.sha256(open(zp,'rb').read()).hexdigest(); open(zp+'.sha256','w').write(f'{sh}  {os.path.basename(zp)}\n'); os.remove(csv); return sh
def mk(name,sc):
    assert sc.min()>=0 and sc.max()<=1 and np.isfinite(sc).all()
    csv=f'out/{name}.csv'; pl.DataFrame({'rowid':D['rowid'],'predicted_probability':sc}).write_csv(csv,float_precision=9); print(name,zipit(csv),'spearman vs EB',round(spearmanr(sc,r['eb'])[0],5),'vs ours',round(spearmanr(sc,r['ours'])[0],5))
for w in (90,80,70): mk(f'probe_eb_v21list_B_w{w}',w/100*r['eb']+(1-w/100)*r['ours'])
mk('probe_eb_v21list_p33_B_w75_20_05',0.75*r['eb']+0.2*r['ours']+0.05*r['p33_rank'])
