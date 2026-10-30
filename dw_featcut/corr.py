import sys; sys.path.insert(0,'.')
from common import *
import pandas as pd, time
t0=time.time(); S=sample(); print(S.shape, round(time.time()-t0),'s',flush=True)
S.write_parquet('sample300k.parquet')
df=S.to_pandas(); del S
nanshare=df.isna().mean(); nun=df.nunique()
top=df.apply(lambda s: s.value_counts(dropna=False,normalize=True).iloc[0])
P=df.corr(method='pearson',min_periods=1000)
R=df.rank(method='average',na_option='keep'); Sp=R.corr(method='pearson',min_periods=1000)
out={'nan_share':nanshare.to_dict(),'nunique':nun.to_dict(),'top_value_share':top.to_dict()}
pairs=[]
cs=list(df.columns)
Pv=P.to_numpy(); Sv=Sp.to_numpy()
for i in range(len(cs)):
    for j in range(i+1,len(cs)):
        p,s=Pv[i,j],Sv[i,j]
        if (abs(p)>0.95 if p==p else False) or (abs(s)>0.95 if s==s else False): pairs.append((cs[i],cs[j],None if p!=p else round(float(p),4),None if s!=s else round(float(s),4)))
out['pairs']=pairs
json.dump(out,open('corr_out.json','w'))
print(len(pairs),'pairs >0.95;', sum(1 for a,b,p,s in pairs if max(abs(p or 0),abs(s or 0))>0.98),'pairs >0.98', round(time.time()-t0),'s')
