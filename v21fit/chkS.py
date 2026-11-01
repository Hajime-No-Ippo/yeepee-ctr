import glob,numpy as np,polars as pl
from lib import *
from scipy.stats import spearmanr
import sys
sys.path.insert(0,'/workspace/huawei/v21seq')
def tologit(lp): lp=np.minimum(lp,-1e-6); return lp-np.log1p(-np.exp(lp))
rid=pl.read_parquet(H+'v12/feat_v3/eval.parquet',columns=['rowid'])['rowid'].to_list(); mrow=M['rowid'].to_list()
fs=sorted(glob.glob('ens/EBDISTILL_A_c27_s*.npy')); S=np.mean([tologit(np.load(f)[:,-1]) for f in fs],0)
Sm=pl.DataFrame({'rowid':rid,'s':S}).join(pl.DataFrame({'rowid':mrow}),on='rowid',how='right',maintain_order='right')['s'].to_numpy()
Sr=nr(Sm); np.save('S_ebdistill.npy',Sr); A=np.load('A_all.npy')
print(len(fs),'seeds; rho(S,EN)',spearmanr(Sr[::5],EN[::5])[0],'rho(S,A)',spearmanr(Sr[::5],A[::5])[0],'rho(A,EN)',spearmanr(A[::5],EN[::5])[0])
