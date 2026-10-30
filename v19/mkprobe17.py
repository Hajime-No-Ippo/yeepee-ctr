"""mkprobe17.py NAME: zips out/NAME.csv (+sha256) and builds the 0.8 rank(model) + 0.2 rank(p33) probe (same method as v12x/out/probes/make_p33_probes.py)."""
import numpy as np, pandas as pd, zipfile, hashlib, os, sys
from scipy.stats import rankdata, spearmanr
H='/workspace/huawei'; O=f'{H}/v19/out'; name=sys.argv[1]
q=pd.read_parquet(f'{H}/teammate/work33/p33_recovered_ranks.parquet'); ref=pd.read_csv(f'{H}/v5/submission_v5.csv').rowid
def zipit(csv):
    zp=csv+'.zip'; zf=zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED); zf.write(csv,arcname=os.path.basename(csv)); zf.close()
    sh=hashlib.sha256(open(zp,'rb').read()).hexdigest(); open(zp+'.sha256','w').write(f'{sh}  {os.path.basename(csv)}.zip\n'); return zp,sh
v=pd.read_csv(f'{O}/{name}.csv'); print(name,'sha',zipit(f'{O}/{name}.csv'))
assert v.rowid.is_unique and set(v.rowid)==set(q.rowid)==set(ref) and len(v)==529390 and v.predicted_probability.notna().all()
qq=q.set_index('rowid').loc[v.rowid]; a=v.predicted_probability.values; b=qq.p33_rank.values; n=len(a)
s=np.clip(0.8*(rankdata(a)-1)/(n-1)+0.2*(rankdata(b)-1)/(n-1),0,1)
df=pd.DataFrame({'rowid':v.rowid,'predicted_probability':s}).set_index('rowid').loc[ref].reset_index()
assert list(df.rowid)==list(ref) and df.predicted_probability.between(0,1).all()
pn='probe_'+name.replace('submission_','')+'_p33_w80'; df.to_csv(f'{O}/{pn}.csv',index=False,float_format='%.9g'); print(pn,zipit(f'{O}/{pn}.csv'))
base=pd.read_csv(f'{H}/v12x/out/submission_v12x_blend.csv').set_index('rowid').predicted_probability
print('spearman vs v12x_blend',spearmanr(a,base.loc[v.rowid].values)[0])
