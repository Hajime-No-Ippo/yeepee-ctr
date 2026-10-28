import numpy as np, pandas as pd, polars as pl
R='/workspace/huawei/v12x/v8regen'
rid12=pl.read_parquet('/workspace/huawei/v12/feat_v3/eval.parquet',columns=['rowid'])['rowid'].to_numpy()
rid8=np.load(R+'/xgb8_eval_rowid.npy'); 
pos=pd.Series(np.arange(len(rid8)),index=rid8).loc[rid12].values
out={}
base=np.load(R+'/work/ens/base_c27_s0.npy'); slow=np.load(R+'/work/ens/slow_c27_s0.npy'); print(base.shape,slow.shape)
out['v8 base LGB (regen)']=base[:,1][pos].astype(float); out['v8 slow LGB (regen)']=slow[:,2][pos].astype(float)
out['v8 LGB mean base+slow']=(out['v8 base LGB (regen)']+out['v8 slow LGB (regen)'])/2
s=pd.read_csv(R+'/submission_v8_regen.csv').set_index('rowid').predicted_probability; out['v8 final (regen, incl. seq)']=s.loc[rid12].values
out['v8-feature XGB (450 rds, negfrac .5)']=np.load(R+'/xgb8_eval.npy')[pos].astype(float)
np.savez('/workspace/huawei/v12x/fp/v8_members.npz',**out)
