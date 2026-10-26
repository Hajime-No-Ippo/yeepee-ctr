import numpy as np, polars as pl, json
from scipy.stats import spearmanr
m=pl.read_parquet('cache_mx/eval_93841df7c9f9.parquet',columns=['rowid'])
sub=pl.read_csv('/workspace/huawei/v2work/out/submission_v8.csv',schema_overrides={'rowid':pl.Utf8})
sq=pl.read_parquet('../feat_sq/eval.parquet',columns=['rowid','sq_logit'])
print(m['rowid'].dtype, sub['rowid'].dtype, sq['rowid'].dtype, m['rowid'][:2].to_list())
j=m.join(sub,on='rowid',how='left',maintain_order='left').join(sq.with_columns(pl.col('rowid').cast(pl.Utf8)),on='rowid',how='left',maintain_order='left')
assert j['predicted_probability'].null_count()==0 and j['sq_logit'].null_count()==0
p8=j['predicted_probability'].to_numpy().astype(float); L8=np.log(p8/(1-p8)); s=j['sq_logit'].to_numpy().astype(float)
z=(np.load('ens/ctl_b_c27_s0.npy')[:,1]+np.load('ens/ctl_s_c27_s0.npy')[:,2])/2; lr=np.load('ens/lr_c27.npy').astype(float)
r=dict(corr_L8_recon=np.corrcoef(L8,.8*z+.2*s)[0,1], corr_L8_z=np.corrcoef(L8,z)[0,1], corr_z_sq=np.corrcoef(z,s)[0,1], std_z=z.std(), std_sq=s.std(),
       implied_sq_corr=np.corrcoef((L8-.8*z)/.2,s)[0,1], max_abs_recon_diff=np.abs(L8-(.8*z+.2*s)).max(), corr_z_lr=np.corrcoef(z,lr)[0,1], corr_L8_lr=np.corrcoef(L8,lr)[0,1], std_lr=lr.std(),
       mean_p8=p8.mean(), mean_sig_sq=(1/(1+np.exp(-s))).mean(), mean_sig_z=(1/(1+np.exp(-z))).mean())
print(json.dumps({k:round(float(v),5) for k,v in r.items()},indent=0))
np.save('/tmp/eval_join_rowid_ok.npy',np.array([1]))
