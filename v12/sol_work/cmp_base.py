import sys, json, numpy as np, polars as pl
W=sys.argv[1]; sys.path.insert(0,W); import os; os.chdir(W)
from cl import matrix, BASE
X,m,fl=matrix('eval',1.0,extra=BASE)
ref='/workspace/huawei/v9d/val/cache_mx/eval_93841df7c9f9.parquet'; rfl=json.load(open(ref+'.json'))
print('ncols sol',len(fl),'ref',len(rfl),'same list',fl==rfl, 'same set',set(fl)==set(rfl))
r=pl.read_parquet(ref); print('rows',len(r),m.height,'rowid same order',(r['rowid'].to_numpy()==m['rowid'].to_numpy()).all())
R=r.select(rfl).to_numpy().astype(np.float32); idx=[fl.index(c) for c in rfl] if set(fl)==set(rfl) else None
Xs=X[:,idx]; nanm=np.isnan(R)!=np.isnan(Xs); print('nan-pattern mismatches cols',int((nanm.sum(0)>0).sum()))
d=np.nanmax(np.abs(R-Xs),0); sc=np.nanmax(np.abs(R),0)+1e-9; rel=d/sc
bad=np.argsort(-rel)[:12]; print('worst rel diffs',[(rfl[i],float(rel[i])) for i in bad]); print('cols with rel>1e-4:',int((rel>1e-4).sum()),'rel>1e-2',int((rel>1e-2).sum()))
fr=np.nanmean(np.abs(R-Xs)>1e-3*(np.abs(R)+1e-3),0); print('cols with >0.1% of values differing (rel 1e-3):',int((fr>1e-3).sum()),'; max frac',float(np.nanmax(fr)))
for i in np.argsort(-fr)[:10]: print(rfl[i],round(float(fr[i]),5), 'corr',round(float(np.corrcoef(np.nan_to_num(R[:,i]),np.nan_to_num(Xs[:,i]))[0,1]),6))
