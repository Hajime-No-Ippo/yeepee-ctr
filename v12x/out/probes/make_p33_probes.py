# Same method as /workspace/huawei/probes/make_probe.py, parametrised by base + weight.
import numpy as np, pandas as pd, zipfile, hashlib, os, sys
from scipy.stats import rankdata, spearmanr
H='/workspace/huawei'; O=f'{H}/v12x/out/probes'
q=pd.read_parquet(f'{H}/teammate/work33/p33_recovered_ranks.parquet')
ref=pd.read_csv(f'{H}/v5/submission_v5.csv').rowid
existing=pd.read_csv(f'{H}/probes/probe_v12_p33_w80.csv').set_index('rowid').predicted_probability
def build(base_csv,w,name,outdir):
    v=pd.read_csv(base_csv)
    assert v.rowid.is_unique and q.rowid.is_unique and set(v.rowid)==set(q.rowid)==set(ref) and len(v)==529390
    qq=q.set_index('rowid').loc[v.rowid]
    a=v.predicted_probability.values; b=qq.p33_rank.values; n=len(a)
    ra=(rankdata(a,'average')-1)/(n-1); rb=(rankdata(b,'average')-1)/(n-1)
    s=np.clip(w*ra+(1-w)*rb,0,1); assert np.isfinite(s).all()
    df=pd.DataFrame({'rowid':v.rowid,'predicted_probability':s}).set_index('rowid').loc[ref].reset_index()
    assert list(df.rowid)==list(ref) and df.predicted_probability.between(0,1).all()
    os.makedirs(outdir,exist_ok=True); csv=f'{outdir}/{name}.csv'
    df.to_csv(csv,index=False,float_format='%.9g')
    zp=csv+'.zip'; zf=zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED); zf.write(csv,arcname=os.path.basename(csv)); zf.close()
    sh=hashlib.sha256(open(zp,'rb').read()).hexdigest(); open(zp+'.sha256','w').write(f'{sh}  {name}.csv.zip\n')
    return df,zp,sh
# verification
df,zp,sh=build(f'{H}/v12/out/submission_v12.csv',0.8,'probe_v12_p33_w80',f'/tmp/verify_w80')
ex=existing.loc[df.rowid].values
print('VERIFY csv bytes equal:',open('/tmp/verify_w80/probe_v12_p33_w80.csv','rb').read()==open(f'{H}/probes/probe_v12_p33_w80.csv','rb').read())
print('VERIFY max abs diff',np.abs(df.predicted_probability.values-ex).max(),'spearman',spearmanr(df.predicted_probability.values,ex)[0])
print('VERIFY zip sha rebuilt',sh,'existing',open(f'{H}/probes/probe_v12_p33_w80.csv.zip.sha256').read().split()[0])
for w,name in ((0.7,'probe_v12x_p33_w70'),(0.8,'probe_v12x_p33_w80')):
    df,zp,sh=build(f'{H}/v12x/out/submission_v12x_blend.csv',w,name,O)
    sp=spearmanr(df.predicted_probability.values,existing.loc[df.rowid].values)[0]
    base=pd.read_csv(f'{H}/v12x/out/submission_v12x_blend.csv').set_index('rowid').predicted_probability
    sb=spearmanr(df.predicted_probability.values,base.loc[df.rowid].values)[0]
    print(name,len(df),'NaN',int(df.predicted_probability.isna().sum()),'sha256',sh,'spearman vs probe_v12_p33_w80',round(sp,6),'vs v12x_blend',round(sb,6),'range',df.predicted_probability.min(),df.predicted_probability.max())
