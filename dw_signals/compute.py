import sys, polars as pl
from signals import *
from sklearn.metrics import roc_auc_score as A
c=int(sys.argv[1]); X=Jd(c); D=Data(); V=D.val(c); S=D.snap(X); log('snap',len(S['cu']))
Vclk,Uc=click_svd(S,D,32); log('click svd')
F={}
def add(d,tag):
    F.update(d); log('done',tag,list(d))
add(sig_skip_embedding(D,S,V,X,Vclk)[0],'skip')
add(sig_slate_centrality(D,S,V,X,Vclk,'clk')[0],'centrality_clk')
add(sig_exposure_lift(D,S,V,X)[0],'expo')
add(sig_user_time(D,S,V,X)[0],'usertime')
o,ex=sig_impression_svd(D,S,V,X); add(o,'impsvd')
add(sig_slate_centrality(D,S,V,X,ex['Vi']/ (np.linalg.norm(ex['Vi'],axis=1,keepdims=True)+1e-9),'imp')[0],'centrality_imp')
add(sig_coimpression(D,S,V,X)[0],'coimp')
pl.DataFrame({k:np.asarray(v,np.float32) for k,v in F.items()}).write_parquet(f'cache/sig_C{c}.parquet'); log('saved')
