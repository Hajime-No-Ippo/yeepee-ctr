import polars as pl, numpy as np, lightgbm as lgb, sys
from sklearn.metrics import roc_auc_score
sys.path.insert(0,'/tmp/dig'); from h import wauc
R=pl.read_parquet('rows3.parquet')
E3=pl.read_parquet('ev3.parquet').select('eid','prev_y','since_hclick','since_prev_ev','gap_med','ctr_prev')
R=R.join(E3,on='eid',how='left')
D=[]
for c in (25,26):
    d=R.filter(pl.col(f'v{c}')==1).join(pl.read_parquet(f'base_{c}.parquet'),on='rowid').join(pl.read_parquet(f'F_val{c}.parquet'),on='rowid')
    D.append(d.with_columns(pl.lit(c).alias('cut')))
D=pl.concat(D,how='diagonal_relaxed')
NEW=['slate_c','slate_s','uimp_c','uimp_s','uh_s','uh_c','prev_y','since_hclick','since_prev_ev','gap_med']
import json; old=list(json.load(open('/tmp/imp_s0.json')))
# nearest existing feature
Xo=D.select(old).to_pandas().rank()
for f in NEW:
    s=D[f].to_pandas().rank(); cr=Xo.corrwith(s).abs().sort_values(ascending=False)
    print(f'{f:14s} nearest {cr.index[0]} |rho|={cr.iloc[0]:.3f}  2nd {cr.index[1]} {cr.iloc[1]:.3f}')
D.write_parquet('D.parquet')
