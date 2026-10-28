import numpy as np, polars as pl
from sklearn.metrics import roc_auc_score as A
H='/workspace/huawei'
def z12(c):
    s=lambda t,k:np.load(f'{H}/v12/ens/{t}_c{c}_s0.npy')[:,k]
    m=np.mean([np.load(f'{H}/v12/ens/dwmlp_c{c}_s{k}.npy')[:,3] for k in (range(3) if c!=26 else [0])],0)
    return ((s('dwbig',1)+s('dw_s',1)+2*m)/4).astype(np.float64)
def meta(c):
    return pl.read_parquet(f'{H}/v10/feat_v3/val{c}.parquet',columns=['rowid','eid','u','ts','clicked','tm_stale'])
def stale_hours(c):
    return None
def fastauc(y,s):
    return A(y,s)
