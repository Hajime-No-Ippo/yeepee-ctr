"""Local best base logits (read-only on other dirs): base = .4*z + .6*M, M = 8-seed mean of ctl6 (v21seq, var A), z = v19 stack (C25)/dwbig (C23).
Returns logits aligned with v10/feat_v3/val{c}.parquet rows (== v12/feat_v3 order, asserted)."""
import sys; sys.dont_write_bytecode=True
sys.path.insert(0,'/workspace/huawei/v21seq')
import numpy as np, polars as pl
from evalseq import zbase,tologit,AUC
H='/workspace/huawei'
def base_logit(c,nseed=8,wz=.4):
    z=zbase('A',c).astype(np.float64)
    M=tologit(np.mean([np.load(f'{H}/v21seq/ens/ctl6_A_c{c}_s{k}.npy')[:,3] for k in range(nseed)],0)).astype(np.float64)
    return wz*z+(1-wz)*M, z, M
if __name__=='__main__':
    a=pl.read_parquet(f'{H}/v12/feat_v3/val25.parquet',columns=['rowid'])['rowid'].to_numpy(); b=pl.read_parquet(f'{H}/v10/feat_v3/val25.parquet',columns=['rowid'])['rowid'].to_numpy(); print((a==b).all())
    for c in (23,25):
        y=pl.read_parquet(f'{H}/v10/feat_v3/val{c}.parquet',columns=['clicked'])['clicked'].to_numpy()
        B,z,M=base_logit(c); print(c,AUC(y,B),AUC(y,z),AUC(y,M))
