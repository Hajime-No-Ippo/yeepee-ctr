import numpy as np, polars as pl
from reusable import *
D=Data(); V=D.val(25); df=all_recommended(D,Jd(25),V); print(df.shape)
ref=pl.concat([pl.read_parquet(f'cache/{f}_C25.parquet') for f in ('sig','sig2','sig3')],how='horizontal')
bad=[]
for k in df.columns:
    if k in ref.columns:
        a=df[k].to_numpy(); b=ref[k].to_numpy(); m=~(np.isnan(a)|np.isnan(b))
        if (np.isnan(a)!=np.isnan(b)).any() or np.abs(a[m]-b[m]).max()>1e-3: bad.append((k,float(np.abs(a[m]-b[m]).max())))
    else: print('new col',k)
print('mismatch',bad)
