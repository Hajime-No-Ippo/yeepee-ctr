"""Shared loader + feature-matrix cache (step C). Matrices cached as parquet keyed by hash(table names, feature list, sample)."""
import os, sys, hashlib, json, gc, numpy as np, polars as pl
sys.path.insert(0,'@@WORK@@'); sys.path.insert(1,'@@WORK@@/v3')
from feats import DROP
from features import feature_cols, GROUPS
D='@@WORK@@'; P2=f'{D}/feat'; P3=f'{D}/feat_v3'; CACHE=f'{D}/cache_mx'; os.makedirs(CACHE,exist_ok=True)
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
BASE=('feat_lr2','feat_tt','feat_pp')
def matrix(n, frac=1.0, variant='notime', extra=()):
    """v12: base = v8-lite matrix (v2/v3/x + feat_lr2/feat_tt/feat_pp, 'notime'), cached as parquet; every other dir in `extra`
    (feat_sqC, feat_dw) is left-joined on rowid and its columns appended. returns (X float32, meta DataFrame, feats)."""
    base=tuple(e for e in extra if e in BASE); new=[e for e in extra if e not in BASE]; assert base==BASE, extra
    key=hashlib.md5(json.dumps([n,variant,list(base),[os.path.getmtime(f) for f in (f'{P3}/{n}.parquet',f'{P2}/{V2_OF[n]}.parquet',f'{D}/feat_x/{n}.parquet')+tuple(f'{D}/{b}/{n}.parquet' for b in base)]]).encode()).hexdigest()[:12]
    fp=f'{CACHE}/{n}_{key}.parquet'
    if os.path.exists(fp): t=pl.read_parquet(fp); fl=json.load(open(fp+'.json'))
    else:
        t=load_tab(n,base,variant); fl=featlist(t,variant)
        t=t.select(list(dict.fromkeys([c for c in KEEP if c in t.columns]+fl)))
        t=t.with_columns([pl.col(c).cast(pl.Float32) for c in fl if t[c].dtype!=pl.Float32 and c not in KEEP])
        t.write_parquet(fp+'.tmp'); os.replace(fp+'.tmp',fp); json.dump(fl,open(fp+'.json','w'))
    for sub in new:
        e=pl.read_parquet(f'{D}/{sub}/{n}.parquet'); nc=[c for c in e.columns if c!='rowid' and c not in fl]
        t=t.join(e.select(['rowid']+nc),on='rowid',how='left',maintain_order='left'); fl=fl+nc
    X=t.select(fl).cast(pl.Float32).to_numpy(); meta=t.select([c for c in KEEP if c in t.columns]); del t; gc.collect()
    return X, meta, fl
