import sys, polars as pl
from signals import *
c=int(sys.argv[1]); X=Jd(c); D=Data(); V=D.val(c); S=D.snap(X)
F=sig_impression_multirank(D,S,V,X)[0]
pl.DataFrame({k:np.asarray(v,np.float32) for k,v in F.items()}).write_parquet(f'cache/sig3_C{c}.parquet'); log('saved',list(F))
