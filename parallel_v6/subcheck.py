import json, hashlib, numpy as np, polars as pl
from scipy.stats import spearmanr
D='/workspace/huawei/v2work'
s=pl.read_csv(f'{D}/out/submission_v6.csv',schema_overrides={'rowid':pl.Utf8})
e=pl.read_parquet(f'{D}/eval_pred_v6.parquet').with_columns(pl.col('rowid').cast(pl.Utf8))
c=pl.read_parquet(f'{D}/cache_mx/eval_bc6b041e949f.parquet',columns=['rowid'])
b=np.mean([np.load(f'{D}/ens/v6base_c27_s{k}.npy')[:,1] for k in (0,1)],0); sl=np.load(f'{D}/ens/v6slow_c27_s0.npy')[:,2]
p=1/(1+np.exp(-(b+sl)/2)); r=pl.DataFrame({'rowid':c['rowid'].cast(pl.Utf8),'pre':p})
j=s.join(e,on='rowid',how='left').join(r,on='rowid',how='left')
v3=pl.read_csv(f'{D}/out/submission_v3.csv',schema_overrides={'rowid':pl.Utf8}).rename({'predicted_probability':'v3'})
j=j.join(v3,on='rowid',how='left')
out={'sha256':hashlib.sha256(open(f'{D}/out/submission_v6.csv','rb').read()).hexdigest(),'rows':s.height,'unique':s['rowid'].n_unique(),
 'nulls_eval_pred':int(j['p'].null_count()),'maxabs_sub_vs_eval_pred_v6':float((j['predicted_probability']-j['p']).abs().max()),
 'maxabs_sub_vs_recomputed_blend_c27':float((j['predicted_probability']-j['pre']).abs().max()),
 'min':float(s['predicted_probability'].min()),'max':float(s['predicted_probability'].max()),'mean':float(s['predicted_probability'].mean()),
 'spearman_vs_v2work_submission_v3':float(spearmanr(j['predicted_probability'].to_numpy(),j['v3'].to_numpy()).statistic)}
print(json.dumps(out,indent=1)); json.dump(out,open('subcheck.json','w'),indent=1)
