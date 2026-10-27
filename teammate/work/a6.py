import numpy as np, polars as pl
D='/workspace/huawei/v3/data/'
x=pl.read_parquet('X2.parquet')
et=pl.read_parquet(D+'ev_train.parquet',columns=['eid','ts']); rt=pl.read_parquet(D+'rows_train.parquet',columns=['eid','ii','clicked']).with_row_index('o')
rt=rt.join(et,on='eid')
rt=rt.with_columns(pos=pl.col('o')-pl.col('o').min().over('eid'))
print('train CTR by position:', rt.group_by('pos').agg(pl.col('clicked').mean()).sort('pos')['clicked'].round(4).to_list())
T1=1737936000-86400; T2=1737936000-2*86400
r1=rt.filter(pl.col('ts')>=T1).group_by('ii').agg(n1=pl.len()); r2=rt.filter(pl.col('ts')>=T2).group_by('ii').agg(n2=pl.len())
r6=rt.filter(pl.col('ts')>=1737936000-6*3600).group_by('ii').agg(n6h=pl.len())
x=x.join(r1,on='ii',how='left').join(r2,on='ii',how='left').join(r6,on='ii',how='left').with_columns(pl.col(['n1','n2','n6h']).fill_null(0))
x.select('ord','n1','n2','n6h').write_parquet('Xrecent.parquet')
cols=['it_n','n2','n1','n6h','tr_ihist','pos','ev_icand','ev_ihist']
rng=np.random.default_rng(0)
for seg,f in [('cold',pl.col('hlen')==0),('warm',pl.col('hlen')>0)]:
    w=x.filter(f).with_columns([((pl.col(c).rank()-5.5)/2.87).over('eid').alias('r'+c) for c in ['p3','v8']+cols])
    if seg=='warm': w=w.filter(pl.col('eid')%10==0)
    X=w.select(['r'+c for c in cols]).to_numpy(); e=w['eid'].to_numpy(); ue=np.unique(e); idx={u:np.where(e==u)[0] for u in ue}
    print(seg, 'events', len(ue), 'corr(ev_icand, n1)=%.3f corr(ev_icand,it_n)=%.3f'%(np.corrcoef(w['rev_icand'],w['rn1'])[0,1],np.corrcoef(w['rev_icand'],w['rit_n'])[0,1]))
    for t in ['rp3','rv8']:
        y=w[t].to_numpy(); beta=np.linalg.lstsq(X,y,rcond=None)[0]; bs=[]
        for _ in range(200):
            s=np.concatenate([idx[u] for u in rng.choice(ue,len(ue))]); bs.append(np.linalg.lstsq(X[s],y[s],rcond=None)[0])
        bs=np.array(bs); print('  ',t,' | '.join(f'{c} {b:+.3f}[{lo:+.3f},{hi:+.3f}]' for c,b,lo,hi in zip(cols,beta,np.percentile(bs,2.5,0),np.percentile(bs,97.5,0))))
