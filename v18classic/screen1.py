"""single-feature screening: global AUC and within-event AUC (click-events only: P(clicked cand > random non-clicked cand in same slate), ties .5) + nan%. python screen1.py CUTNAME"""
import sys, numpy as np, polars as pl, json
from sklearn.metrics import roc_auc_score
nm=sys.argv[1]; y=pl.read_parquet(f'/workspace/huawei/v12/feat_v3/{nm}.parquet',columns=['clicked'])['clicked'].to_numpy(); Y=y.reshape(-1,10); has=Y.sum(1)==1
res={}
for f in ('fr','au','uc','ty'):
    d=pl.read_parquet(f'/workspace/huawei/v18classic/feat_{f}/{nm}.parquet')
    for c in d.columns[1:]:
        x=d[c].to_numpy().astype(np.float64); nanp=np.isnan(x).mean(); xx=np.nan_to_num(x,nan=np.nanmedian(x) if nanp<1 else 0); 
        try: g=roc_auc_score(y,xx)
        except Exception: g=np.nan
        M=xx.reshape(-1,10)[has]; yy=Y[has].astype(bool); xc=M[yy]; lt=((M<xc[:,None]).sum(1)+0.5*((M==xc[:,None]).sum(1)-1))/9; w=lt.mean()
        res[c]=(f,round(nanp*100,1),round(g,4),round(w,4)); print(f'{f} {c:22s} nan%={nanp*100:5.1f} globalAUC={g:.4f} withinAUC={w:.4f}')
json.dump(res,open(f'/workspace/huawei/v18classic/logs/screen1_{nm}.json','w'))
