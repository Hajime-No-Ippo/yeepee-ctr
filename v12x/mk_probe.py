import numpy as np, pandas as pd, zipfile, hashlib, os
from scipy.stats import spearmanr
V='/workspace/huawei/v12/ens'; X='/workspace/huawei/v12x/ens'; O='/workspace/huawei/v12x/out'; os.makedirs(O,exist_ok=True)
import polars as pl
rid=pl.read_parquet('/workspace/huawei/v12/feat_v3/eval.parquet',columns=['rowid'])['rowid'].to_numpy()
ref=pd.read_csv('/workspace/huawei/data/evaluation.csv',usecols=['rowid']).rowid
ld=lambda p,c: np.load(p)[:,c].astype(np.float64)
big=ld(f'{V}/dwbig_c27_s0.npy',1); sm=ld(f'{V}/dw_s_c27_s0.npy',1); ml=np.mean([ld(f'{V}/dwmlp_c27_s%d.npy'%s,3) for s in range(3)],0); xg=ld(f'{X}/xgb_c27_s0.npy',1)
sig=lambda z:1/(1+np.exp(-z))
v12=pd.read_csv('/workspace/huawei/v12/out/submission_v12.csv').set_index('rowid').predicted_probability
for name,z in (('submission_v12x_xgb_only',xg),('submission_v12x_blend',(big+sm+2*ml+xg)/5)):
    s=pd.Series(np.clip(sig(z),0,1),index=rid); df=pd.DataFrame({'rowid':ref,'predicted_probability':s.loc[ref].values})
    assert len(df)==529390 and df.rowid.is_unique and np.isfinite(df.predicted_probability).all() and df.predicted_probability.between(0,1).all() and set(df.rowid)==set(rid)
    csv=f'{O}/{name}.csv'; df.to_csv(csv,index=False,float_format='%.9g')
    zp=csv+'.zip'; zf=zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED); zf.write(csv,arcname=os.path.basename(csv)); zf.close()
    sh=hashlib.sha256(open(zp,'rb').read()).hexdigest(); open(zp+'.sha256','w').write(f'{sh}  {os.path.basename(zp)}\n')
    print(name,len(df),'NaN',int(df.predicted_probability.isna().sum()),'sha256',sh,'spearman vs v12',round(spearmanr(df.predicted_probability,v12.loc[df.rowid].values)[0],5),'range',df.predicted_probability.min(),df.predicted_probability.max())
