import sys, json, numpy as np, polars as pl, itertools, os
sys.path.insert(0,'/workspace/huawei/v12x'); sys.path.insert(1,'/workspace/huawei/v12x/pylib')
from sklearn.metrics import roc_auc_score
V='/workspace/huawei/v12/ens'; X='/workspace/huawei/v12x/ens'
y={c:pl.read_parquet(f'/workspace/huawei/v12/feat_v3/val{c}.parquet',columns=['clicked'])['clicked'].to_numpy() for c in (25,23)}
def ld(p,col=0): return np.load(p)[:,col].astype(np.float64)
def auc(c,s): return round(roc_auc_score(y[c],s),5)
def base(c):
    big=ld(f'{V}/dwbig_c{c}_s0.npy',1); sm=ld(f'{V}/dw_s_c{c}_s0.npy',1) if c==25 else ld(f'{V}/dw_s_c{c}_s0.npy',1)
    ml=np.mean([ld(f'{V}/dwmlp_c{c}_s{s}.npy',3) for s in (0,1,2)],0); return big,sm,ml
def demean(s): g=s.reshape(-1,10); return (g-g.mean(1,keepdims=True)).ravel()
if __name__=='__main__':
    for c in (25,23):
        big,sm,ml=base(c); z=(big+sm+2*ml)/4; print('cut',c,'dwbig',auc(c,big),'dw_s',auc(c,sm),'mlp',auc(c,ml),'v12 blend',auc(c,z))
