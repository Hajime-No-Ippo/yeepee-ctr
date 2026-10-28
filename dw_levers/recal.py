import sys, numpy as np, polars as pl, pandas as pd
sys.path.insert(0,'/workspace/huawei/dw_levers')
from common import z12, A
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
H='/workspace/huawei'
for c in (25,23):
    m=pl.read_parquet(f'{H}/v10/feat_v3/val{c}.parquet',columns=['clicked','u','us_hlen','it_imp','us_nev','us_cold']).to_pandas()
    z=z12(c); y=m.clicked.values.astype(int); base=A(y,z)
    for name,keys in (('hlen',['us_hlen']),('imp',['it_imp']),('hlen x imp',['us_hlen','it_imp']),('nev x imp',['us_nev','it_imp'])):
        cell=np.zeros(len(y),int)
        for k in keys: cell=cell*8+pd.qcut(m[k],6,duplicates='drop',labels=False).values.astype(int)
        out=np.zeros(len(y))
        for tr,te in GroupKFold(5).split(z,y,m.u.values):   # honest CV by user
            for cl in np.unique(cell):
                a=tr[cell[tr]==cl]; b=te[cell[te]==cl]
                if len(b)==0: continue
                lr=LogisticRegression(C=10).fit(z[a,None],y[a]); out[b]=lr.decision_function(z[b,None])
        print(f'C{c} {name:12s} Δ={A(y,out)-base:+.5f}',flush=True)
