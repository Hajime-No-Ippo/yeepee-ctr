import numpy as np, polars as pl
from scipy.stats import rankdata
D=pl.read_parquet('../merged.parquet'); n=D.height
nr=lambda x:(rankdata(x)-1)/(n-1)
EN=np.load('eb_new_rank.npy')
D.with_columns(pl.Series('en',EN),pl.Series('e0',nr(D['eb'].to_numpy())),pl.Series('o',nr(D['ours'].to_numpy())),pl.Series('p',nr(D['p33_rank'].to_numpy()))).with_columns((pl.col('en')-pl.col('e0')).alias('d')).select(['event_id','rowid','e0','en','o','p','d']).write_parquet('cmp.parquet')
