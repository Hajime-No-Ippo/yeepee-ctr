import sys; sys.path.insert(0,'/workspace/huawei/v21seq')
import numpy as np,polars as pl,zipfile,hashlib,os
from lib import *
from scipy.stats import spearmanr
H0='/workspace/huawei'
rid=pl.read_parquet(f'{H0}/v12/feat_v3/eval.parquet',columns=['rowid'])['rowid'].to_list(); mrow=M['rowid'].to_list()
def ordered(vals,order): return pl.DataFrame({'rowid':rid,'p':vals}).join(pl.DataFrame({'rowid':order}),on='rowid',how='right',maintain_order='right')['p'].to_numpy()
def tologit(lp): lp=np.minimum(lp,-1e-6); return lp-np.log1p(-np.exp(lp))
cols=[(a,int(b),c) for a,b,c in (x.split(':') for x in sys.argv[1:])]  # tag:col:seed
D=np.mean([tologit(np.load(f'ens/{t}_A_c27_s{s}.npy')[:,c]) for t,c,s in cols],0); D=nr(ordered(D,mrow)); np.save('D_dcn.npy',D)
A=np.load('A_all.npy'); print('rho D,A',np.corrcoef(D,A)[0,1],'rho D,EN',spearmanr(D[::3],EN[::3])[0],'rho A,EN',spearmanr(A[::3],EN[::3])[0])
