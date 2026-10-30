import os; os.environ.setdefault('OMP_NUM_THREADS','1'); os.environ.setdefault('POLARS_MAX_THREADS','1'); os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json, glob, re, numpy as np, polars as pl, pyarrow.parquet as pq
H='/workspace/huawei'
REF=json.load(open(f'{H}/v20/feats_ref484.json'))   # 484 production-candidate columns, matrix order (read-only copy of list)
def cols(p): return [c for c in pq.read_schema(p).names if c!='rowid']
DIRS={'feat_dw':f'{H}/v12/feat_dw','feat_Ap':f'{H}/v17cooc/feat_Ap','feat_NAs':f'{H}/v19/feat_NAs','feat_sqC':f'{H}/v11/feat_sqC',
      'feat_tt':f'{H}/v11/feat_tt','feat_pp':f'{H}/v11/feat_pp','feat_lr2':f'{H}/v11/feat_lr2','feat_x':f'{H}/v11/feat_x'}
def src_map():
    m={}
    for k,d in DIRS.items():
        for c in cols(f'{d}/val25.parquet'):
            if c in REF and c not in m: m[c]=k
    v3=set(cols(f'{H}/v11/feat_v3/val25.parquet'))
    for c in REF:
        if c not in m: m[c]='feat_v3' if c in v3 else ('v2_' if c.startswith('v2_') else '??')
    return m
def sample(tables=('f2','f4'),n=150000,seed=0):
    """base cache (v8-lite 377 cols) + extra dirs joined on rowid, subsample rows. Training-fold tables only (no eval rows)."""
    rs=np.random.RandomState(seed); parts=[]
    for t in tables:
        fp=glob.glob(f'{H}/v9d/val/cache_mx/{t}_*.parquet'); fp=[p for p in fp if p.endswith('.parquet')]; assert len(fp)==1
        base=pl.read_parquet(fp[0]); idx=np.sort(rs.choice(base.height,n,replace=False)); b=base[idx]; del base
        for k,d in DIRS.items():
            if k in ('feat_lr2','feat_tt','feat_pp','feat_x'): continue   # already in base cache
            e=pl.read_parquet(f'{d}/{t}.parquet'); nc=[c for c in e.columns if c!='rowid' and c not in b.columns]
            b=b.join(e.select(['rowid']+nc),on='rowid',how='left',maintain_order='left')
        parts.append(b.select(REF).cast(pl.Float32))
    return pl.concat(parts)
