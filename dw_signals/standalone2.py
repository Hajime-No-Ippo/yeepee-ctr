import sys, polars as pl, numpy as np
from common import *
from sklearn.metrics import roc_auc_score as A
c=int(sys.argv[1]); D=Data(); V=D.val(c); y=V['y']; sg=segs(c); nh=sg['nohist']==1
F=pl.concat([pl.read_parquet(f'cache/sig{s}_C{c}.parquet') for s in sys.argv[2].split(',')],how='horizontal'); ev=np.repeat(np.arange(len(y)//10),10)
for k in F.columns:
    x=F[k].to_numpy().astype(float); m=~np.isnan(x)
    if m.sum()<1000: continue
    xx=np.nan_to_num(x,nan=np.nanmedian(x)); 
    # within-event auc
    z=within_event(xx)[0]
    print(f'{k:26s} cov={m.mean():.3f} glob={A(y,xx):.4f} within={A(y,z):.4f} nohist_rows={A(y[nh],xx[nh]):.4f}')
