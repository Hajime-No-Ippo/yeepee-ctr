"""Shared loader + feature-matrix cache (step C). Matrices cached as parquet keyed by hash(table names, feature list, sample)."""
import os, sys, hashlib, json, gc, numpy as np, polars as pl
sys.path.insert(0,'/workspace/huawei/v9'); sys.path.insert(1,'/workspace/huawei/v3')
from feats import DROP
from features import feature_cols, GROUPS
V9='/workspace/huawei/v9'; D='/workspace/huawei/v10'; P2=f'{V9}/feat'; P3=f'{V9}/feat_v3'; CACHE='/workspace/huawei/v9d/val/cache_mx'  # read-only base cache
V2_OF={'val23':'val23','f1':'h1','f2':'h2','f3':'h3','f4':'h4','f5':'h5','f6':'val26','val25':'val25','val26':'val26','eval':'eval'}
V2_OF.update({f'{p}{k}':f'{p}{k}' for p in 'eEd' for k in range(1,7)})
TIME=['tm_stale','tm_hour','tm_tb','tm_dt','v2_tb','v2_dt','v2_stale','v2_hour']
KEEP=['rowid','eid','u','ts','clicked','us_cold','us_nohist','tm_stale']
def load_tab(n, extra=(), variant='notime'):
    a=pl.read_parquet(f'{P3}/{n}.parquet'); b=pl.read_parquet(f'{P2}/{V2_OF[n]}.parquet')
    f2=[c for c in b.columns if c not in DROP]
    a=a.join(b.select(['rowid']+f2).rename({c:'v2_'+c for c in f2}),on='rowid',how='left')
    for sub in (os.environ.get('XDIR','feat_x'),)+(('feat_x2',) if os.path.isdir(f'{D}/feat_x2') else ())+tuple(extra):
        a=a.join(pl.read_parquet(f'{D}/{sub}/{n}.parquet'),on='rowid',how='left')
    if '+' in variant:
        import mods; a=mods.apply(a,n,variant)
    return a
def featlist(df, variant='notime'):
    f=feature_cols(df,GROUPS)+[c for c in df.columns if c.startswith(('v2_','x_','r_','s_','ev2_','rc_','sh_','hs_','lr','tA_','tB_','uw_','la_','sq_'))]
    if 'notime' in variant: f=[c for c in f if c not in TIME]
    if variant.split('+')[0]=='stale': f=[c for c in f if c not in TIME or c=='tm_stale']
    return f
def usample(df, frac):
    if frac>=1: return df
    return df.filter((pl.col('u').hash(seed=7) % 10000) < int(frac*10000))
FL=[None]
def load_tab_rows(n, rid, extra=(), variant='notime'):
    """same as load_tab but restricted to rowids rid (polars Series) -> low memory"""
    a=pl.scan_parquet(f'{P3}/{n}.parquet').filter(pl.col('rowid').is_in(rid)).collect()
    b=pl.scan_parquet(f'{P2}/{V2_OF[n]}.parquet').filter(pl.col('rowid').is_in(rid)).collect()
    f2=[c for c in b.columns if c not in DROP]
    a=a.join(b.select(['rowid']+f2).rename({c:'v2_'+c for c in f2}),on='rowid',how='left')
    for sub in (os.environ.get('XDIR','feat_x'),)+(('feat_x2',) if os.path.isdir(f'{D}/feat_x2') else ())+tuple(extra):
        a=a.join(pl.scan_parquet(f'{D}/{sub}/{n}.parquet').filter(pl.col('rowid').is_in(rid)).collect(),on='rowid',how='left')
    return a
def build_chunked(n, extra, variant, frac, nch=4):
    rid_all=pl.read_parquet(f'{P3}/{n}.parquet',columns=['rowid'])['rowid']; parts=[]
    step=(len(rid_all)+nch-1)//nch
    for i in range(nch):
        rid=rid_all.slice(i*step,step); t=usample(load_tab_rows(n,rid,extra,variant),frac)
        # restore original order within chunk
        t=pl.DataFrame({'rowid':rid}).join(t,on='rowid',how='inner',maintain_order='left')
        fl=featlist(t,variant); FL[0]=fl
        t=t.select(list(dict.fromkeys([c for c in KEEP if c in t.columns]+fl)))
        t=t.with_columns([pl.col(c).cast(pl.Float32) for c in fl if t[c].dtype!=pl.Float32 and c not in KEEP])
        parts.append(t); gc.collect()
    return pl.concat(parts)
BASE=('feat_lr2','feat_tt','feat_pp')
def matrix(n, frac=1.0, variant='notime', extra=()):
    """v9: base = cached v8-lite matrix (read-only, from v9d/val/cache_mx); new dirs (feat_tm*) joined on rowid.
    returns (X float32, meta DataFrame, feats)."""
    base=tuple(e for e in extra if e in BASE); new=[e for e in extra if e not in BASE]; assert base==BASE, extra
    key=hashlib.md5(json.dumps([n,frac,variant,list(base),os.path.getmtime(f'{P3}/{n}.parquet')]).encode()).hexdigest()[:12]
    fp=f'{CACHE}/{n}_{key}.parquet'; assert os.path.exists(fp), fp
    fl=json.load(open(fp+'.json')); t=pl.read_parquet(fp)
    for sub in new:
        e=pl.read_parquet(f'{D}/{sub}/{n}.parquet'); nc=[c for c in e.columns if c!='rowid' and c not in fl]
        t=t.join(e.select(['rowid']+nc),on='rowid',how='left',maintain_order='left'); fl=fl+nc
    drop=os.environ.get('V9DROP')
    if drop:
        import re; fl=[c for c in fl if not re.search(drop,c)]
    X=t.select(fl).cast(pl.Float32).to_numpy(); meta=t.select([c for c in KEEP if c in t.columns]); del t; gc.collect()
    return X, meta, fl
