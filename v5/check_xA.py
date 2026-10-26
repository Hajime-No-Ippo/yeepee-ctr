import polars as pl, numpy as np, sys, json
new=None
for n in 'f1 f2 f3 f4 f5 f6 val23 val25 val26 eval'.split():
    A=pl.read_parquet(f'feat_xA/{n}.parquet'); O=pl.read_parquet(f'feat_x/{n}.parquet', columns=['rowid'])
    nc=[c for c in A.columns if c not in pl.read_parquet_schema(f'feat_x/{n}.parquet')]
    gone=[c for c in pl.read_parquet_schema(f'feat_x/{n}.parquet') if c not in A.columns]
    X=A.select(nc).to_numpy().astype(float)
    bad=int((~np.isfinite(X)).sum())
    same = A.height==O.height and set(A['rowid'].to_list())==set(O['rowid'].to_list())
    print(n, A.height, 'rows_match', same, 'new_cols', len(nc), 'removed', gone, 'nonfinite', bad)
    assert same and bad==0
    new=nc
print('NEWCOLS', json.dumps(new))
