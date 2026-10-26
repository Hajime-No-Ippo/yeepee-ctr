"""Rank-average of submission_v1 (external percentile ranks) and submission_v2. Not measurable on local validation.
Usage: python blend.py [w_v2=0.5]"""
import sys, numpy as np, polars as pl
w = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
rd = lambda p: pl.read_csv(p, schema_overrides={'rowid': pl.Utf8, 'predicted_probability': pl.Float64})
ex = rd('/workspace/huawei/example_submission.csv').select('rowid')
v1 = rd('/workspace/huawei/submission_v1.csv'); v2 = rd('/workspace/huawei/submission_v2.csv')
J = ex.join(v1.rename({'predicted_probability': 'p1'}), on='rowid', how='left', maintain_order='left') \
      .join(v2.rename({'predicted_probability': 'p2'}), on='rowid', how='left', maintain_order='left')
assert J.height == ex.height and J['p1'].null_count() == 0 and J['p2'].null_count() == 0
n = J.height
r1 = J['p1'].rank('average').to_numpy() / n; r2 = J['p2'].rank('average').to_numpy() / n
p = np.clip(np.nan_to_num((1 - w) * r1 + w * r2, nan=0.0629), 0, 1)
out = '/workspace/huawei/v3/submission_blend.csv'
pl.DataFrame({'rowid': J['rowid'], 'predicted_probability': p}).to_pandas().to_csv(out, index=False, float_format='%.8g')
print('wrote', out, 'w_v2', w)
