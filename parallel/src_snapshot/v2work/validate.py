"""Standalone validator: python validate.py SUB.csv"""
import sys, csv, math, numpy as np, polars as pl
out=sys.argv[1]
ids=pl.read_csv('/workspace/huawei/v2work/evaluation.csv',columns=['rowid'],schema_overrides={'rowid':pl.Utf8})['rowid'].to_list()
raw=open(out,'rb').read(); errs=[]
try: raw.decode('utf-8')
except Exception: errs.append('not utf-8')
with open(out,newline='',encoding='utf-8') as f:
    r=csv.reader(f); h=next(r); rows=list(r)
if h!=['rowid','predicted_probability']: errs.append(f'header {h}')
if len(rows)!=529390: errs.append(f'rows {len(rows)}')
sid=[x[0] for x in rows]
if len(set(sid))!=len(sid): errs.append('dup rowids')
if set(sid)!=set(ids): errs.append('rowid set mismatch')
v=np.array([float(x[1]) for x in rows])
if not (np.isfinite(v).all() and (v>=0).all() and (v<=1).all()): errs.append('bad values')
print(f'rows={len(rows)} unique={len(set(sid))} min={v.min():.4g} mean={v.mean():.5g} max={v.max():.4g} distinct={len(np.unique(v))}')
print('VALID' if not errs else 'INVALID '+';'.join(errs))
