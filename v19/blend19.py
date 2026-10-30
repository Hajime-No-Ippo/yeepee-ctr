"""blend19.py: C25 numbers. python blend19.py  -> prints component + blend AUC for v17cooc (co) vs v19 nm (pruned NMF) for A,B and any pairwise members present"""
import sys, os, numpy as np, polars as pl
from sklearn.metrics import roc_auc_score as AUC
from scipy.stats import rankdata
H='/workspace/huawei'; y=pl.read_parquet(f'{H}/v12/feat_v3/val25.parquet',columns=['clicked'])['clicked'].to_numpy()
def comp(v,kind,c=25):
    if kind=='co': E=f'{H}/v17cooc/ens'; big,s,m=f'co{v}p_c{c}_s0',f'co{v}_s_c{c}_s0',f'co{v}mlp_c{c}_s%d'
    else: E=f'{H}/v19/ens'; big,s,m=f'nm{v}s_c{c}_s0',f'nm{v}s_s_c{c}_s0',f'nm{v}smlp_c{c}_s%d'
    L=lambda f,col:np.load(f'{E}/{f}.npy')[:,col]
    b=L(big,0); d=L(s,1); ml=np.mean([L(m%k,3) for k in (0,1,2)],0); return b,d,ml,(b+d+2*ml)/4
if __name__=='__main__':
    for v in 'BA':
        for kind in ('co','nm'):
            try: b,d,ml,z=comp(v,kind); print(v,kind,'dwbig %.5f dw_s %.5f mlp(3) %.5f blend %.5f'%tuple(AUC(y,t) for t in (b,d,ml,z)))
            except Exception as e: print(v,kind,'missing',str(e)[:60])
