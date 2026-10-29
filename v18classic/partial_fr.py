"""Does freshness add beyond popularity? val25. (1) within-event AUC of freshness feature after residualising its within-event-demeaned value on within-event-demeaned popularity features;
(2) within-event AUC of a logistic regression (2-fold by event) with popularity-only vs popularity+freshness (demeaned within event)."""
import glob, numpy as np, polars as pl
from sklearn.linear_model import LogisticRegression
POP=['it_ctr','it_ctr24','v2_ip_n','v2_i_ctr','it_nclk_all','it_nusers','it_imp','v2_ip_dec1','it_clk_dec2d']
fp=glob.glob('/workspace/huawei/v9d/val/cache_mx/val25_*.parquet')[0]
P=pl.read_parquet(fp,columns=['rowid','clicked']+POP); F=pl.read_parquet('/workspace/huawei/v18classic/feat_fr/val25.parquet')
d=P.join(F,on='rowid',how='left',maintain_order='left'); y=d['clicked'].to_numpy().reshape(-1,10); has=y.sum(1)==1
FR=['fr_age_h','fr_gap_imp_h','fr_gap_clk_h','fr_gap_clk_rel','fr_age_norm']
def dm(c): x=d[c].to_numpy().astype(np.float64).reshape(-1,10); x=np.nan_to_num(x,nan=np.nanmedian(x)); return (x-x.mean(1,keepdims=True))
def wauc(s):
    M=s[has]; yy=y[has].astype(bool); xc=M[yy]; return (((M<xc[:,None]).sum(1)+0.5*((M==xc[:,None]).sum(1)-1))/9).mean()
Pm=np.stack([dm(c).ravel() for c in POP],1); Pm=(Pm-Pm.mean(0))/Pm.std(0)
print('popularity feats',POP)
for c in FR:
    f=dm(c).ravel(); f=(f-f.mean())/f.std(); b=np.linalg.lstsq(Pm,f,rcond=None)[0]; res=(f-Pm@b).reshape(-1,10)
    print(f'{c:16s} raw within-AUC {wauc(f.reshape(-1,10)):.4f} | corr with pop-fit R2 {1-res.var()/f.var():.3f} | residual within-AUC {wauc(res):.4f}')
Fm=np.stack([dm(c).ravel() for c in FR],1); Fm=(Fm-Fm.mean(0))/Fm.std(0)
def cv(X):
    n=len(y); ev=np.arange(n); fold=ev%2; out=np.zeros(n*10)
    Xr=X.reshape(n,10,-1); yy=y.reshape(-1)
    for k in (0,1):
        tr=np.repeat(fold!=k,10); te=np.repeat(fold==k,10)
        m=LogisticRegression(C=1.0,max_iter=300).fit(X[tr],yy[tr]); out[te]=m.decision_function(X[te])
    return wauc(out.reshape(-1,10))
a=cv(Pm); b=cv(np.hstack([Pm,Fm])); print(f'logistic within-AUC pop-only {a:.4f}  pop+freshness {b:.4f}  delta {b-a:+.4f}')
Fl=Fm[:,[1,2]]; print('pop + last-imp/last-click gaps only', round(cv(np.hstack([Pm,Fl])),4))
