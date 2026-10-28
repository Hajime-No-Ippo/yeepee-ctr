import numpy as np, pandas as pd, zipfile, hashlib
from scipy.stats import rankdata, spearmanr
v=pd.read_csv('/workspace/huawei/v12/out/submission_v12.csv')
q=pd.read_parquet('/workspace/huawei/teammate/work33/p33_recovered_ranks.parquet')
ref=pd.read_csv('/workspace/huawei/v5/submission_v5.csv').rowid   # same rowid set/order as v8 (v8 csv lost in restore)
assert v.rowid.is_unique and q.rowid.is_unique and set(v.rowid)==set(q.rowid)==set(ref) and len(v)==529390
q=q.set_index('rowid').loc[v.rowid]
a=v.predicted_probability.values; b=q.p33_rank.values
n=len(a); ra=(rankdata(a,'average')-1)/(n-1); rb=(rankdata(b,'average')-1)/(n-1)
print('spearman v12 vs p33',spearmanr(a,b)[0],'pearson of ranks',np.corrcoef(ra,rb)[0,1])
s=0.8*ra+0.2*rb; print('score range',s.min(),s.max(),'finite',np.isfinite(s).all(),'ties',n-len(np.unique(s)))
s=np.clip(s,0,1)
df=pd.DataFrame({'rowid':v.rowid,'predicted_probability':s}); df=df.set_index('rowid').loc[ref].reset_index()
assert list(df.rowid)==list(ref) and df.predicted_probability.between(0,1).all()
df.to_csv('/workspace/huawei/probes/probe_v12_p33_w80.csv',index=False,float_format='%.9g')
zf=zipfile.ZipFile('/workspace/huawei/probes/probe_v12_p33_w80.csv.zip','w',zipfile.ZIP_DEFLATED); zf.write('/workspace/huawei/probes/probe_v12_p33_w80.csv',arcname='probe_v12_p33_w80.csv'); zf.close()
print(hashlib.sha256(open('/workspace/huawei/probes/probe_v12_p33_w80.csv.zip','rb').read()).hexdigest())
print('spearman probe vs v12',spearmanr(df.predicted_probability,v.set_index('rowid').predicted_probability.loc[df.rowid])[0])
