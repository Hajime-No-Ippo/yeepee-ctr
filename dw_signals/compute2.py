import sys, polars as pl
from signals import *
c=int(sys.argv[1]); X=Jd(c); D=Data(); V=D.val(c); S=D.snap(X)
Vclk,Uc=click_svd(S,D,32); F={}
F.update(sig_impression_variants(D,S,V,X)[0]); log('variants')
F.update(sig_recent_impression(D,S,V,X,Vclk)[0]); log('recent')
pl.DataFrame({k:np.asarray(v,np.float32) for k,v in F.items()}).write_parquet(f'cache/sig2_C{c}.parquet'); log('saved',list(F))
