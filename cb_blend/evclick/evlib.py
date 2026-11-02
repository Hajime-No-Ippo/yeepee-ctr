import os, sys, glob, json, re, gc, numpy as np, polars as pl
os.environ.setdefault('OMP_NUM_THREADS','4'); os.environ.setdefault('POLARS_MAX_THREADS','4')
H='/workspace/huawei'; CACHE=f'{H}/v9d/val/cache_mx'
SUBS=['feat_sqC','feat_dw','feat_Ap','feat_NAs']
def sdir(sub):
    for b in (f'{H}/v19',f'{H}/v17cooc',f'{H}/v12x',f'{H}/v12'):
        if os.path.isdir(f'{b}/{sub}'): return f'{b}/{sub}'
def files(n):
    fp=glob.glob(f'{CACHE}/{n}_*.parquet'); assert len(fp)==1,(n,fp)
    return [fp[0]]+[f'{sdir(s)}/{n}.parquet' for s in SUBS]
_SCH={}
def where(n):
    if n not in _SCH:
        d={}
        for f in files(n):
            for c in pl.scan_parquet(f).collect_schema().names():
                d.setdefault(c,f)
        _SCH[n]=d
    return _SCH[n]
def cols(n, names, meta=('rowid','eid','u','ts','clicked','tm_stale','us_cold','us_nohist')):
    """meta + requested columns for table n, base-cache row order; float32."""
    w=where(n); by={}
    for c in list(meta)+list(names):
        if c in w: by.setdefault(w[c],[]).append(c)
    fs=files(n); base=fs[0]; df=pl.read_parquet(base,columns=sorted(set(by.get(base,[]))|{'rowid'}))
    for f in fs[1:]:
        if f in by:
            e=pl.read_parquet(f,columns=sorted(set(by[f])|{'rowid'}))
            df=df.join(e,on='rowid',how='left',maintain_order='left')
    miss=[c for c in names if c not in df.columns]
    return df,miss
