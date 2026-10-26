import polars as pl, numpy as np, hashlib, json
from scipy.stats import spearmanr, pearsonr
a=pl.read_csv('/workspace/huawei/v2work/out/submission_v3.csv',schema_overrides={'rowid':pl.Utf8})
b=pl.read_csv('/workspace/huawei/submission_v2.csv',schema_overrides={'rowid':pl.Utf8})
e=pl.read_parquet('/workspace/huawei/v2work/eval_pred_v3.parquet').with_columns(pl.col('rowid').cast(pl.Utf8))
o=pl.read_parquet('/workspace/huawei/v2work/eval_pred.parquet').with_columns(pl.col('rowid').cast(pl.Utf8))
j=a.join(b,on='rowid',suffix='_ours').join(e,on='rowid',how='left').join(o.rename({'p':'p_old'}),on='rowid',how='left')
r={'rows_v2work_sub':a.height,'rows_ours':b.height,'rows_joined':j.height,
   'sha256_v2work_sub':hashlib.sha256(open('/workspace/huawei/v2work/out/submission_v3.csv','rb').read()).hexdigest(),
   'max_abs_diff_sub_vs_eval_pred_v3':float((j['predicted_probability']-j['p']).abs().max()),
   'max_abs_diff_sub_vs_old_eval_pred':float((j['predicted_probability']-j['p_old']).abs().max()),
   'spearman_v2work_v3sub_vs_our_submission_v2':float(spearmanr(j['predicted_probability'],j['predicted_probability_ours']).statistic),
   'pearson':float(pearsonr(j['predicted_probability'],j['predicted_probability_ours']).statistic),
   'mean_v2work':float(j['predicted_probability'].mean()),'mean_ours':float(j['predicted_probability_ours'].mean())}
print(json.dumps(r,indent=1)); json.dump(r,open('/workspace/huawei/parallel/subcheck.json','w'),indent=1)
