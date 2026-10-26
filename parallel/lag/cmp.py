import polars as pl, numpy as np, json
V='/workspace/huawei/v2work'; L='/workspace/huawei/parallel/lag'; out={}
for d,a,b in (('feat_v3','val25','val25repro'),('feat','val25','val25repro'),('feat_x','val25','val25'),('feat_x2','val25','val25')):
    A=pl.read_parquet(f'{V}/{d}/{a}.parquet'); B=pl.read_parquet(f'{L}/{d}/{b}.parquet')
    B=A.select('rowid').join(B,on='rowid',how='left')
    cols=[c for c in A.columns if c in B.columns and c!='rowid' and A[c].dtype.is_numeric()]
    bad={}
    for c in cols:
        x=A[c].cast(pl.Float64).to_numpy(); y=B[c].cast(pl.Float64).to_numpy()
        dif=np.nanmax(np.abs(x-y)) if np.isfinite(x-y).any() else 0.0
        nanm=int((np.isnan(x)!=np.isnan(y)).sum())
        if dif>1e-3*max(1,np.nanmax(np.abs(x))) or nanm: bad[c]=(float(dif),nanm)
    out[d]=dict(ncols=len(cols),only_disk=[c for c in A.columns if c not in B.columns],n_diff=len(bad),diff=dict(list(bad.items())[:12]))
print(json.dumps(out,indent=1)); json.dump(out,open(f'{L}/repro_cmp.json','w'),indent=1)
