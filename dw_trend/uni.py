import sys; sys.path.insert(0,'/workspace/huawei/dw_trend')
from run_trend import *
for c in (25,23):
    F,df=load(c); y=F['y']; z=F['z']; n=len(y)//10; Y=y.reshape(n,10); out=[]
    for k in ITEM+USER+TASTE+TT:
        a=df[k].to_numpy().astype(np.float64).reshape(n,10); a=a-np.nanmean(a,1,keepdims=True); a=np.nan_to_num(a)
        ok=(Y.sum(1)>0)&(np.abs(a).sum(1)>0); 
        out.append((k,round(A(Y[ok].ravel(),a[ok].ravel()),4) if ok.sum()>100 else "event-level"))
        # partial: correlation with base residual not needed
    print(c,out)
