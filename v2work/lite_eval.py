import numpy as np, os
from cl import matrix
from sklearn.metrics import roc_auc_score as A
L=lambda t,c,s,ci: np.load(f'ens/{t}_c{c}_s{s}.npy')[:,ci]
for c in (25,26):
    _,m,_=matrix(f'val{c}',1.0,extra=('feat_lr2','feat_tt')); y=m['clicked'].to_numpy()
    b0,b1,sl=L('v7base',c,0,1),L('v7base',c,1,1),L('v7slow',c,0,2)
    r={'full_v7(2base+slow)':A(y,((b0+b1)/2+sl)/2),'a:1base+slow':A(y,(b0+sl)/2),
       'cmb_base+cmb_slow':A(y,(L('lt_cmb_base',c,0,1)+L('lt_cmb_slow',c,0,2))/2),
       'cmb_base_only':A(y,L('lt_cmb_base',c,0,1)),'cmb_slow_only':A(y,L('lt_cmb_slow',c,0,2)),
       'v7base_s0':A(y,b0)}
    for t in ('lt_neg50','lt_f3','lt_top175','lt_bin63'):
        r[t+'+v7slow']=A(y,(L(t,c,0,1)+sl)/2)
    print(c,{k:round(v,5) for k,v in r.items()},flush=True)
