"""mksub.py OUT SPEC : spec like cmp11 (tag:col items joined by '+', optional w*), cut 27 -> sigmoid(mean score) -> OUT csv (rowid,p) ordered as v8 reference."""
import sys, numpy as np, polars as pl, pandas as pd
from cl import matrix
out, spec = sys.argv[1], sys.argv[2]
from cl import BASE
_, m, _ = matrix('eval', 1.0, extra=BASE)
def load(item):
    w=1.0
    if '*' in item: w,item=item.split('*'); w=float(w)
    p=item.split(':'); sd=p[2].split(',') if len(p)>2 else ['0']; return w*np.mean([np.load(f'ens/{p[0]}_c27_s{x}.npy')[:,int(p[1])] for x in sd],0)
it=spec.split('+'); W=sum(float(i.split('*')[0]) if '*' in i else 1.0 for i in it); z=np.sum([load(i) for i in it],0)/W
assert len(z)==m.height; p=1/(1+np.exp(-z)); assert np.isfinite(p).all() and (p>=0).all() and (p<=1).all()
df=pd.DataFrame({'rowid':m['rowid'].to_numpy(),'p':p})
ref=pd.read_csv('/workspace/huawei/v2work/out/submission_v8.csv'); print(ref.columns.tolist(), len(ref))
assert df.rowid.is_unique and len(df)==529390 and set(df.rowid)==set(ref.iloc[:,0]), 'rowid set mismatch'
df=df.set_index('rowid').loc[ref.iloc[:,0]].reset_index(); df.columns=list(ref.columns)
df.to_csv(out,index=False); print('wrote',out,len(df),'p range',p.min(),p.max(),'mean',p.mean())
from scipy.stats import spearmanr
v10=pd.read_csv('/workspace/huawei/v10/out/submission_v10.csv'); a=df.set_index(df.columns[0]).iloc[:,0]; b=v10.set_index(v10.columns[0]).iloc[:,0].loc[a.index]
print('spearman vs v10', spearmanr(a,b)[0]); v8=ref.set_index(ref.columns[0]).iloc[:,0].loc[a.index]; print('spearman vs v8', spearmanr(a,v8)[0])
