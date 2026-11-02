"""helpers: to_merged(x_v21seq_order)->array in merged order; rho(x) vs EN etc (sample 200k, pearson of ranks like fitg)"""
import sys,os,numpy as np,polars as pl; sys.dont_write_bytecode=True
sys.path.insert(0,'/workspace/huawei/v21fit'); os.chdir('/workspace/huawei/v21graph2')
from scipy.stats import rankdata
H='/workspace/huawei/'; O='/workspace/huawei/v21graph2/out/'
mrow=pl.read_parquet(H+'erbrown/merged.parquet',columns=['rowid'])['rowid'].to_list()
EN=np.load(H+'erbrown/w10an/eb_new_rank.npy'); rid=np.load(O+'tab_eval.npz')['rowid'].astype(str)
nr=lambda x:(rankdata(x)-1)/(len(x)-1)
def to_merged(x):
    gm=pl.DataFrame({'rowid':rid,'p':np.asarray(x,np.float64)}).join(pl.DataFrame({'rowid':mrow}),on='rowid',how='right',maintain_order='right')['p'].to_numpy(); assert np.isfinite(gm).all(); return gm
idx=np.random.RandomState(0).choice(len(EN),200000,replace=False)
G7=np.load('/workspace/huawei/v21graph/out/graph_eval_merged_rank.npy').astype(np.float64)
def rho(xm,ref=EN): return np.corrcoef(nr(xm)[idx],ref[idx])[0,1]
