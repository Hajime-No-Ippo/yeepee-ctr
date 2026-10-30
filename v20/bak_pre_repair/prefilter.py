"""Step 1 (no training; light, 1 thread): near-constant + |Spearman|>0.98 prefilter on a 300k-row sample of the C25 training fold."""
import os; os.environ.update(OMP_NUM_THREADS='1',POLARS_MAX_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys; sys.path.insert(0,'/workspace/huawei/v20')
import json, numpy as np, pandas as pd, re
from lib20 import *
fl=json.load(open('/workspace/huawei/v20/feats_ref484.json'))
rs=np.random.RandomState(0); parts=[]
for n in trn_tables(25):
    X,m,f=matrix(n,1.0,variant='notime',extra=EXTRA); ix=[f.index(c) for c in fl]; sel=np.sort(rs.choice(len(X),75000,replace=False)); parts.append(X[sel][:,ix]); del X,m
S=np.concatenate(parts); del parts; print(S.shape,flush=True)
gr=json.load(open('gain_rank.json'))['ranked']; gain={d['feature']:d['gain_norm_mean'] for d in gr}
const={}; 
for j,c in enumerate(fl):
    v=S[:,j]; vv=v[~np.isnan(v)]
    if len(vv)==0: const[c]='all-NaN'; continue
    _,cnt=np.unique(np.concatenate([vv,np.full(len(v)-len(vv),np.inf)]),return_counts=True)
    if cnt.max()/len(v)>0.999: const[c]=f'{cnt.max()/len(v):.5f} share one value'
alive=[c for c in fl if c not in const]; ix=[fl.index(c) for c in alive]
R=pd.DataFrame(S[:,ix],columns=alive).rank(method='average',na_option='keep'); R=R.fillna(R.mean()); del S
C=np.corrcoef(R.to_numpy(np.float32).T.astype(np.float64)); C=np.nan_to_num(C)
order=sorted(alive,key=lambda c:-gain[c]); idx={c:i for i,c in enumerate(alive)}; dropped={}; kept=[]
for c in order:
    if c in dropped: continue
    kept.append(c)
    for d in order:
        if d!=c and d not in dropped and d not in kept and abs(C[idx[c],idx[d]])>0.98: dropped[d]=dict(kept_in_favour_of=c,abs_spearman=round(float(abs(C[idx[c],idx[d]])),4))
clusters={}
for d,v in dropped.items(): clusters.setdefault(v['kept_in_favour_of'],[]).append(d)
json.dump(dict(n_in=len(fl),near_constant=const,n_dropped_redundant=len(dropped),n_clusters=len(clusters),n_kept=len(kept),kept=kept,dropped=dropped,clusters=clusters,
  method='rank (NaN->mean rank) Pearson on ranks = Spearman; greedy by descending mean gain: keep representative, drop later cols with |rho|>0.98 vs a kept one'),open('out/redundancy.json','w'),indent=1)
print('const',len(const),'redundant dropped',len(dropped),'clusters',len(clusters),'kept',len(kept))
