"""OPTIONAL p33 probe: 0.8 * rank(our submission) + 0.2 * rank(p33).  python mkprobe.py SUBMISSION.csv P33_PATH OUT.csv
P33_PATH = parquet with columns rowid, p33_rank (teammate p33's recovered ranks; NOT reproducible from this repo -- supplied by the teammate). Output order = evaluation.csv order."""
import sys, numpy as np, polars as pl, zipfile, hashlib, os
from scipy.stats import rankdata
sub, p33, out = sys.argv[1:4]
v=pl.read_csv(sub,schema_overrides={'rowid':pl.Utf8}); q=pl.read_parquet(p33).select('rowid','p33_rank')
assert v['rowid'].is_unique() and set(v['rowid'])==set(q['rowid']) and v.height==529390
j=v.join(q,on='rowid',how='left',maintain_order='left'); a=j['predicted_probability'].to_numpy(); b=j['p33_rank'].to_numpy(); n=len(a)
s=np.clip(0.8*(rankdata(a)-1)/(n-1)+0.2*(rankdata(b)-1)/(n-1),0,1)
pl.DataFrame({'rowid':j['rowid'],'predicted_probability':s}).write_csv(out,float_precision=9)
zp=out+'.zip'
with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED) as z: z.write(out,arcname=os.path.basename(out))
sh=hashlib.sha256(open(zp,'rb').read()).hexdigest(); open(zp+'.sha256','w').write(f'{sh}  {os.path.basename(zp)}\n'); print('probe',out,sh)
