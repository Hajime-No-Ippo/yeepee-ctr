import sys, numpy as np, polars as pl, json
sys.path.insert(0,'/workspace/huawei/dw_levers')
from common import z12, A
H='/workspace/huawei'
out={}
for c in (25,23):
    m=pl.read_parquet(f'{H}/v10/feat_v3/val{c}.parquet',columns=['rowid','eid','clicked','u','tm_stale','us_hlen','hist_n_win','it_imp','us_nev','tm_hour','it_ctr','hist_in','us_cold','us_nohist']).to_pandas()
    z=z12(c); y=m.clicked.values.astype(int); base=A(y,z); print(f'== C{c} base {base:.5f} n={len(y)} pos={y.mean():.4f}')
    # global AUC when segment scores replaced by oracle-shifted: seg pos get +big? use: pos in seg -> z+3 , neg in seg -> z-3
    def segs(col,bins):
        q=pd_cut(m[col].values,bins); return q
    import pandas as pd
    pd_cut=lambda x,b: pd.cut(x,b,include_lowest=True).astype(str)
    S={'tm_stale':pd_cut(m.tm_stale.values,[-1e9,0,6,12,24,48,1e9]) ,
       'us_hlen':pd.qcut(m.us_hlen,6,duplicates='drop').astype(str),
       'hist_n_win':pd.qcut(m.hist_n_win,6,duplicates='drop').astype(str),
       'it_imp':pd.qcut(m.it_imp,6,duplicates='drop').astype(str),
       'us_nev':pd.qcut(m.us_nev,6,duplicates='drop').astype(str),
       'hour':pd_cut(m.tm_hour.values,[-1,5,11,17,24])}
    for k,sg in S.items():
        print('--',k)
        for s in sorted(set(sg)):
            i=sg==s
            if i.sum()<500 or y[i].min()==y[i].max(): continue
            wi=A(y[i],z[i]); zz=z.copy(); zz[i&(y==1)]+=2; zz[i&(y==0)]-=2   # soft oracle
            # recalibration headroom: shift segment so mean logit gap matches
            hs=A(y,zz)-base
            # mean score (pos rate) vs mean pred
            print(f'  {s:>28} share {i.mean():.3f} pos {y[i].mean():.3f} AUCin {wi:.4f} meanZ {z[i].mean():+.2f} oracleΔ {hs:+.4f} oracleΔ/share {hs/i.mean():+.4f}')
