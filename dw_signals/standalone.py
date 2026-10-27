import sys, polars as pl, numpy as np
from sklearn.metrics import roc_auc_score as A
from common import *
c=int(sys.argv[1]); D=Data(); V=D.val(c); y=V['y']; F=pl.read_parquet(f'cache/sig_C{c}.parquet'); sg=segs(c)
Y=y.reshape(-1,10); has=Y.sum(1)==1; hl=np.repeat(D.Hlen[V['e'][::10]],10)
z=v11_pred(c)
def wea(x):
    X=x.reshape(-1,10)[has]; xc=(X*Y[has]).sum(1); ok=~np.isnan(xc)
    X=X[ok];xc=xc[ok]; Xn=np.nan_to_num(X,nan=np.nanmean(X)); xcn=xc
    return (((Xn<xc[:,None]).sum(1)+0.5*((Xn==xc[:,None]).sum(1)-1))/9).mean()
rows=[]
for k in F.columns:
    x=F[k].to_numpy().astype(float); m=~np.isnan(x); 
    g=A(y[m],x[m]); w=wea(x) if np.isnan(x).mean()<0.5 else np.nan
    ev=x.reshape(-1,10)[:,0]; evlev=np.nan
    if np.nanstd(x.reshape(-1,10),1).max()==0: # event-level feature
        e0=x.reshape(-1,10)[:,0]; mm=~np.isnan(e0); evlev=A(has[mm],e0[mm])
    # residual: corr of x with v11 within event? use global AUC on nohist
    nh=sg['nohist']==1; mh=m&nh
    gnh=A(y[mh],x[mh]) if mh.sum()>2000 else np.nan
    rows.append((k,round(g,4),round(w,4) if w==w else None,round(evlev,4) if evlev==evlev else None,round(gnh,4) if gnh==gnh else None,round(m.mean(),3)))
print(f'C{c} v11 global {A(y,z):.4f} within {wea(z):.4f}')
print('feature | globalAUC | withinEventAUC | eventLevelAUC(has click) | AUC nohist rows | coverage')
for r in rows: print(*r,sep=' | ')
