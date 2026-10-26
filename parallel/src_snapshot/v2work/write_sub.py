"""Write + validate submission from eval_pred.parquet. Usage: python write_sub.py OUT.csv"""
import sys, csv, math, hashlib, zipfile, os, numpy as np, polars as pl
out=sys.argv[1]
pr=pl.read_parquet(os.environ.get('PRED','/workspace/huawei/v2work/eval_pred.parquet'))
ev=pl.read_csv('/workspace/huawei/v2work/evaluation.csv',columns=['rowid'],schema_overrides={'rowid':pl.Utf8})
df=ev.join(pr,on='rowid',how='left',maintain_order='left')
p=np.nan_to_num(df['p'].to_numpy().astype(np.float64),nan=0.0629,posinf=1.0,neginf=0.0).clip(0,1)
p[np.isnan(df['p'].to_numpy().astype(float))]=0.0629
print('missing preds filled:',int(df['p'].null_count()))
with open(out,'w',encoding='utf-8',newline='') as f:
    f.write('rowid,predicted_probability\n'); f.writelines(f'{r},{v:.8g}\n' for r,v in zip(df['rowid'].to_list(),p))
# ---- validator
ids=ev['rowid'].to_list(); errs=[]
with open(out,newline='',encoding='utf-8') as f:
    r=csv.reader(f); h=next(r); rows=list(r)
if h!=['rowid','predicted_probability']: errs.append('header')
if len(rows)!=529390 or len(ids)!=529390: errs.append(f'rows {len(rows)}')
sid=[x[0] for x in rows]
if len(set(sid))!=len(sid) or set(sid)!=set(ids): errs.append('rowid set mismatch')
vals=[float(x[1]) for x in rows]
if not all(math.isfinite(v) and 0<=v<=1 for v in vals): errs.append('bad values')
v=np.array(vals); print(f'rows={len(rows)} unique={len(set(sid))} min={v.min():.4g} mean={v.mean():.4g} max={v.max():.4g} distinct={len(np.unique(v))}')
print('VALID' if not errs else 'INVALID '+';'.join(errs))
zp=out+'.zip'
with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED) as z: z.write(out,arcname=os.path.basename(out))
with zipfile.ZipFile(zp) as z: print('zip contents',z.namelist())
print('sha256',hashlib.sha256(open(out,'rb').read()).hexdigest())
