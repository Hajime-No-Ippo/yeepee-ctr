import polars as pl, numpy as np
from sklearn.metrics import roc_auc_score
def wauc(df,c,y='clicked',g='eid'):
    d=df.select(g,y,pl.col(c).cast(pl.Float64).fill_null(-1e18).alias('s')).with_columns(pl.col('s').rank('average').over(g).alias('rk'),
        pl.col(y).sum().over(g).alias('P'),pl.len().over(g).alias('N')).filter((pl.col('P')>0)&(pl.col('P')<pl.col('N')))
    e=d.filter(pl.col(y)==1).group_by(g).agg(((pl.col('rk').sum()-pl.col('P').first()*(pl.col('P').first()+1)/2)/(pl.col('P').first()*(pl.col('N').first()-pl.col('P').first()))).alias('a'))
    return float(e['a'].mean())
def gauc(df,c,y='clicked'):
    d=df.select(y,pl.col(c).cast(pl.Float64)).drop_nulls().drop_nans()
    return float(roc_auc_score(d[y].to_numpy(),d[c].to_numpy()))
def eauc(df,c,y='clicked',g='eid'):
    e=df.group_by(g).agg(pl.col(y).max(),pl.col(c).first()).drop_nulls()
    return float(roc_auc_score(e[y].to_numpy(),e[c].cast(pl.Float64).to_numpy()))
def rep(df,c,event_level=False):
    out={}
    for v in ('v25','v26'):
        d=df.filter(pl.col(v)==1)
        a=gauc(d,c); out[v]=dict(glob=round(max(a,1-a),4),dirn='+' if a>=.5 else '-')
        if event_level: e=eauc(d,c); out[v]['event']=round(max(e,1-e),4)
        else: out[v]['within']=round(wauc(d,c),4)
    return out
