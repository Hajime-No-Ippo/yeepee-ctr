import numpy as np, polars as pl, pandas as pd, hashlib
from scipy.stats import spearmanr
ev=pl.read_csv('/workspace/huawei/data/evaluation.csv',columns=['rowid'])['rowid'].to_list()
m=pl.read_parquet('/workspace/huawei/v11/feat_v3/eval.parquet',columns=['rowid'])['rowid'].to_list(); assert m==ev
big=np.load('ens/sCbig_c27_s0.npy'); s=np.load('ens/sC_s_c27_s0.npy'); print(big.shape,s.shape)
mlp=np.mean([np.load(f'ens/mlp2_c27_s{i}.npy')[:,3] for i in range(3)],0)
z=(big[:,1]+s[:,1]+2*mlp)/4; p=1/(1+np.exp(-z)); print('p range',p.min(),p.max())
pd.DataFrame({'rowid':m,'predicted_probability':p}).to_csv('out/submission_v11.csv',index=False)
print(hashlib.sha256(open('out/submission_v11.csv','rb').read()).hexdigest())
v12=pd.read_csv('/workspace/huawei/v12/out/submission_v12.csv'); print('spearman vs v12',spearmanr(p,v12.predicted_probability)[0])
