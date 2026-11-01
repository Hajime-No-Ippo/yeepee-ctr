import numpy as np, polars as pl, zipfile, io, os
from scipy.stats import rankdata
H='/workspace/huawei/'
M=pl.read_parquet(H+'erbrown/merged.parquet'); n=M.height
meta=pl.read_parquet(H+'v21fit/meta.parquet')
assert meta['rowid'].to_list()==M['rowid'].to_list()
nr=lambda x:(rankdata(x)-1)/(len(x)-1)
def rz(p):
    z=zipfile.ZipFile(p); d=pl.read_csv(io.BytesIO(z.read(z.namelist()[0])))
    assert d['rowid'].to_list()==M['rowid'].to_list(); return d['predicted_probability'].to_numpy()
EN=np.load(H+'erbrown/w10an/eb_new_rank.npy'); E0=nr(M['eb'].to_numpy())
O=nr(M['ours'].to_numpy()); P=nr(M['p33_rank'].to_numpy())
def sub(p): return nr(rz(p))
ev=M['event_id'].to_numpy(); codes=np.unique(ev,return_inverse=True)[1]
cnt=np.bincount(codes)
def smean(x): return (np.bincount(codes,x)/cnt)[codes]
hl=meta['hlen'].to_numpy(); st=meta['stale_s'].to_numpy().astype(float)/3600
