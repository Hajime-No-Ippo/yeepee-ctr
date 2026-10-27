import polars as pl, glob, sys, json
for c in (25,23):
    f=glob.glob(f'/workspace/huawei/v9d/val/cache_mx/val{c}_*.parquet')[0]
    cols=['rowid','sv_k64_ucos','mf_score','it_ctr','co_cos_max20','x_clctr64']
    d=pl.read_parquet(f,columns=cols); v=pl.read_parquet(f'/workspace/huawei/v10/feat_v3/val{c}.parquet',columns=['rowid'])
    d=v.join(d,on='rowid',how='left',maintain_order='left').drop('rowid').rename({k:'decoy_'+k for k in cols[1:]})
    d.write_parquet(f'cache/extra_C{c}.parquet'); print(c,d.shape)
