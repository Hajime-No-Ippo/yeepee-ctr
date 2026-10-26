import polars as pl, numpy as np, json
V='/workspace/huawei/v2work'; L='/workspace/huawei/parallel/lag'; d0=json.load(open(f'{L}/repro_cmp.json')); out={}
for d,b in (('feat_v3','val25repro'),('feat','val25repro'),('feat_x','val25')):
    A=pl.read_parquet(f'{V}/{d}/val25.parquet'); B=A.select('rowid').join(pl.read_parquet(f'{L}/{d}/{b}.parquet'),on='rowid',how='left')
    full=json.load(open(f'{L}/repro_cmp.json'))[d]
    import itertools
    cols=[c for c in A.columns if c!='rowid' and c in B.columns and A[c].dtype.is_numeric()]
    r={}
    for c in cols:
        x=A[c].cast(pl.Float64).to_numpy(); y=B[c].cast(pl.Float64).to_numpy(); m=np.isfinite(x)&np.isfinite(y)
        if np.allclose(x[m],y[m],rtol=1e-4,atol=1e-5): continue
        r[c]=dict(corr=round(float(np.corrcoef(x[m],y[m])[0,1]),5),frac_rows_diff=round(float((~np.isclose(x[m],y[m],rtol=1e-3,atol=1e-4)).mean()),4))
    out[d]=dict(n_cols=len(cols),n_cols_diff=len(r),worst=sorted(r.items(),key=lambda kv:kv[1]['corr'])[:8])
print(json.dumps(out,indent=1)); json.dump(out,open(f'{L}/repro_cmp2.json','w'),indent=1)
