import numpy as np, polars as pl, pandas as pd, hashlib
ev=pl.read_csv('/workspace/huawei/data/evaluation.csv',columns=['rowid'])['rowid'].to_list()
m=pl.read_parquet('/workspace/huawei/v12/feat_v3/eval.parquet',columns=['rowid'])['rowid'].to_list()
print(len(ev),len(m),len(set(ev)),set(ev)==set(m),'same order feat_v3 vs evaluation.csv:',ev==m)
L=lambda t,c,s:np.load(f'ens/{t}_c27_s{s}.npy')[:,c]
z=(L('dwbig',1,0)+L('dw_s',1,0)+2*np.mean([L('dwmlp',3,s) for s in (0,1,2)],0))/4
print(np.load('ens/dwbig_c27_s0.npy').shape,np.load('ens/dw_s_c27_s0.npy').shape,np.load('ens/dwmlp_c27_s0.npy').shape)
p=1/(1+np.exp(-z)); df=pd.DataFrame({'rowid':m,'p':p})
v5=pd.read_csv('/workspace/huawei/v5/submission_v5.csv'); print(v5.columns.tolist(), v5.iloc[:,0].tolist()==ev, v5.iloc[:,0].tolist()==m)
for name,order in (('feat_v3',m),('evalcsv',ev)):
    d=df.set_index('rowid').loc[order].reset_index(); d.to_csv(f'recov/sub_{name}.csv',index=False)
    print(name,hashlib.sha256(open(f'recov/sub_{name}.csv','rb').read()).hexdigest())
from scipy.stats import spearmanr
print('spearman vs v5',spearmanr(df.set_index('rowid').p.loc[v5.iloc[:,0]].values,v5.iloc[:,1].values)[0])
