import sys, polars as pl
from sklearn.metrics import roc_auc_score as A
for d in sys.argv[1:]:
    out=[]
    for c in (25,23,26):
        try:
            v=pl.read_parquet(f'/workspace/huawei/v10/feat_v3/val{c}.parquet',columns=['rowid','clicked']).join(pl.read_parquet(f'{d}/val{c}.parquet',columns=['rowid','sq_logit']),on='rowid',how='left')
            out.append(f"C{c} {A(v['clicked'],v['sq_logit']):.5f}")
        except Exception as e: out.append(f"C{c} -")
    print(d, *out, flush=True)
