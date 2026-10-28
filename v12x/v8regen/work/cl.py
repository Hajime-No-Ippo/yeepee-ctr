"""Shared loader + feature-matrix cache (step C). Matrices cached as parquet keyed by hash(table names, feature list, sample)."""
import os, sys, hashlib, json, gc, numpy as np, polars as pl
sys.path.insert(0,'/workspace/huawei/v12x/v8regen/work'); sys.path.insert(1,'/workspace/huawei/v12x/v8regen/work/v3')
from feats import DROP
from features import feature_cols, GROUPS
D='/workspace/huawei/v12x/v8regen/work'; P2=f'{D}/feat'; P3=f'{D}/feat_v3'; CACHE=f'{D}/cache_mx'; os.makedirs(CACHE,exist_ok=True)
V2_OF={'f1':'h1','f2':'h2','f3':'h3','f4':'h4','f5':'h5','f6':'val26','val25':'val25','val26':'val26','eval':'eval'}
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
        raise ValueError('mods not used in v7')
    return a
def featlist(df, variant='notime'):
    f=feature_cols(df,GROUPS)+[c for c in df.columns if c.startswith(('v2_','x_','r_','s_','ev2_','rc_','sh_','hs_','lr','tA_','tB_'))]
    if 'notime' in variant: f=[c for c in f if c not in TIME]
    if variant.split('+')[0]=='stale': f=[c for c in f if c not in TIME or c=='tm_stale']
    return f
def usample(df, frac):
    if frac>=1: return df
    return df.filter((pl.col('u').hash(seed=7) % 10000) < int(frac*10000))
def matrix(n, frac=1.0, variant='notime', extra=()):
    """returns (X float32, meta DataFrame, feats); cached."""
    key=hashlib.md5(json.dumps([n,frac,variant,list(extra),[os.path.getmtime(f) for f in (f'{P3}/{n}.parquet',f'{P2}/{V2_OF[n]}.parquet',f'{D}/feat_x/{n}.parquet')+((f'{D}/feat_x2/{n}.parquet',) if os.path.isdir(f'{D}/feat_x2') else ())]+[os.path.getmtime(f'{D}/{x}/{n}.parquet') for x in extra]]+([os.environ['XDIR']] if os.environ.get('XDIR','feat_x')!='feat_x' else [])).encode()).hexdigest()[:12]
    fp=f'{CACHE}/{n}_{key}.parquet'
    if os.path.exists(fp):
        t=pl.read_parquet(fp); feats=[c for c in t.columns if c not in KEEP or c in ('us_cold','us_nohist')]
        fl=json.load(open(fp+'.json')); 
    else:
        t=usample(load_tab(n,extra,variant),frac); fl=featlist(t,variant)
        t=t.select(list(dict.fromkeys([c for c in KEEP if c in t.columns]+fl)))
        t=t.with_columns([pl.col(c).cast(pl.Float32) for c in fl if t[c].dtype!=pl.Float32 and c not in KEEP])
        t.write_parquet(fp); json.dump(fl,open(fp+'.json','w'))
    X=t.select(fl).to_numpy().astype(np.float32); meta=t.select([c for c in KEEP if c in t.columns]); del t; gc.collect()
    return X, meta, fl
