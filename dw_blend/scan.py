import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import numpy as np, polars as pl, glob, re, csv
from sklearn.metrics import roc_auc_score
H='/workspace/huawei'
Y={c:pl.read_parquet(f'{H}/v12/feat_v3/val{c}.parquet',columns=['clicked'])['clicked'].to_numpy() for c in (23,25)}
N={c:len(Y[c]) for c in Y}; NE=529390
rows=[]
fs=[f for f in glob.glob(f'{H}/**/*.npy',recursive=True) if re.search(r'_c(23|25|27)(_s\d+)?\.npy$',f) or re.search(r'/(val|)(23|25)\.npy$',f)]
for f in sorted(fs):
    m=re.search(r'_c(23|25|27)(?:_s(\d+))?\.npy$',f)
    if not m: continue
    c=int(m.group(1))
    try: a=np.load(f,mmap_mode='r')
    except Exception as e: continue
    n=a.shape[0]; shp='x'.join(map(str,a.shape))
    ok = (c==27 and n==NE) or (c in N and n==N[c])
    aucs=[]
    if c in N and n==N[c]:
        A=np.asarray(a,dtype=np.float64).reshape(n,-1)
        for j in range(A.shape[1]):
            try: aucs.append(round(roc_auc_score(Y[c],A[:,j]),5))
            except: aucs.append(None)
    base=re.sub(r'_c(23|25|27)(_s\d+)?\.npy$','',os.path.relpath(f,H))
    rows.append((base,c,m.group(2) or '',os.path.relpath(f,H),shp,ok,aucs))
with open('scan_raw.csv','w') as g:
    w=csv.writer(g); w.writerow(['key','cut','seed','path','shape','rowcount_ok','col_aucs']); w.writerows(rows)
print(len(rows))
