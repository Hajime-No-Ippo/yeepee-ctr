import numpy as np, polars as pl, zipfile, io
from scipy.stats import rankdata, spearmanr
D=pl.read_parquet('../merged.parquet'); n=D.height
z=zipfile.ZipFile('submission_blend_newbest_v21_w10_20261031.csv.zip'); W=pl.read_csv(io.BytesIO(z.read(z.namelist()[0])))['predicted_probability'].to_numpy()
nr=lambda x:(rankdata(x)-1)/(n-1)
nr2=lambda x:rankdata(x)/(n+1)
nr3=lambda x:rankdata(x)/n
O=D['ours'].to_numpy()
for nm,f in (('(r-1)/(n-1)',nr),('r/(n+1)',nr2),('r/n',nr3)):
    X=f(O)
    A=np.c_[X,np.ones(n)]; 
    # unknown Y: can't regress. Instead test: Y=(W-0.1X)/0.9 uniformity
    Y=(W-0.1*X)/0.9
    print(nm,'Y min max',Y.min(),Y.max(),'mean',Y.mean(),'std',Y.std(),'unif std',1/np.sqrt(12))
    Yr=nr(Y); print('  max|Y-rank(Y)|',np.abs(Y-Yr).max(),'mean abs',np.abs(Y-Yr).mean(), 'Y*(n-1) frac from int',np.abs(Y*(n-1)-np.round(Y*(n-1))).mean())
# scan a
X=nr(O)
for a in (0.08,0.09,0.1,0.11,0.12):
    Y=(W-a*X)/(1-a); print('a',a,'Y min max',Y.min().round(5),Y.max().round(5),'mean',Y.mean().round(6),'std',Y.std().round(6),'max|Y-rank|',np.abs(Y-nr(Y)).max().round(5),'mean',np.abs(Y-nr(Y)).mean().round(6))
# W ties?
print('W unique',len(np.unique(W)),'of',n)
# values sorted check: W*? 
