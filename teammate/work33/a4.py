import numpy as np, polars as pl
x=pl.read_parquet('X33.parquet').join(pl.read_parquet('ranks33.parquet').drop('eid'),on='ord')
cols=['it_n','n2','n1','n6h','tr_ihist','pos','ev_icand','ev_ihist']
rng=np.random.default_rng(0)
for seg,f in [('cold',pl.col('hlen')==0),('warm',pl.col('hlen')>0)]:
    w=x.filter(f)
    if seg=='warm': w=w.filter(pl.col('eid')%10==0)
    w=w.with_columns([((pl.col(c).rank()-5.5)/2.87).over('eid').alias('q'+c) for c in ['rp3','rv8','rp33']+cols])
    X=w.select(['q'+c for c in cols]).to_numpy(); e=w['eid'].to_numpy(); ue=np.unique(e); idx={u:np.where(e==u)[0] for u in ue}
    for t in ['qrp3','qrv8','qrp33']:
        y=w[t].to_numpy(); beta=np.linalg.lstsq(X,y,rcond=None)[0]; bs=[]
        for _ in range(200):
            s=np.concatenate([idx[u] for u in rng.choice(ue,len(ue))]); bs.append(np.linalg.lstsq(X[s],y[s],rcond=None)[0])
        bs=np.array(bs); i=cols.index('ev_icand'); j=cols.index('it_n')
        print(seg,t,'ev_icand %+.3f [%+.3f,%+.3f]'%(beta[i],*np.percentile(bs[:,i],[2.5,97.5])),' it_n %+.3f'%beta[j],' pos %+.3f'%beta[cols.index('pos')])
