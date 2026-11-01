import sys; sys.path.insert(0,'/workspace/huawei/v21seq')
import glob
from evalseq import *
from scipy.stats import rankdata
def dec(a,b):
    A=a.reshape(-1,10); B=b.reshape(-1,10)
    am,bm=A.mean(1),B.mean(1); aw=(A-am[:,None]).ravel(); bw=(B-bm[:,None]).ravel()
    return np.corrcoef(am,bm)[0,1], np.corrcoef(aw,bw)[0,1], am.var(), aw.var()/1 
nr=lambda x:(rankdata(x)-1)/(len(x)-1)
for c in (23,25):
    y=yv(c); z=zbase('A',c); L,_=listp('A',c); fp=sorted(glob.glob(f'/workspace/huawei/v21seq/ens/ctl6_A_c{c}_s*.npy')); M=np.mean([tologit(np.load(f)[:,-1]) for f in fp],0)
    zr,Mr=nr(z),nr(M); print(c,'z vs M rho_between,rho_within,var_b,var_w',dec(zr,Mr))
    def mix(w_src,b_src):
        W=w_src.reshape(-1,10); B=b_src.reshape(-1,10); return (W-W.mean(1,keepdims=True)+B.mean(1,keepdims=True)).ravel()
    print(' AUC z',round(AUC(y,zr),5),'M',round(AUC(y,Mr),5),'z-within+M-slate',round(AUC(y,mix(zr,Mr)),5),'M-within+z-slate',round(AUC(y,mix(Mr,zr)),5))
    # how much AUC due to within vs slate-level
    for nm,x in (('z',zr),('M',Mr)):
        X=x.reshape(-1,10); print(' ',nm,'within-only(pure demeaned)',round(AUC(y,(X-X.mean(1,keepdims=True)).ravel()),5),'slate-mean-only',round(AUC(y,np.repeat(X.mean(1),10)),5))
    for lam in (0,.25,.5,.75,1):
        Z=zr.reshape(-1,10);Mm=Mr.reshape(-1,10); 
        t=(.4*zr+.6*Mr).reshape(-1,10); print('  blend with slate mean shift to M slate mean lam',lam, round(AUC(y,(t-lam*(t.mean(1,keepdims=True)-Mm.mean(1,keepdims=True))).ravel()),5), 'base',round(AUC(y,t.ravel()),5))
import lib as Lb

def dec2(a,b):
    ma,mb=Lb.smean(a),Lb.smean(b); sel=np.arange(len(a))
    first=np.unique(Lb.codes,return_index=True)[1]
    return np.corrcoef(ma[first],mb[first])[0,1], np.corrcoef(a-ma,b-mb)[0,1], ma[first].var(), (a-ma).var()
for nm,o in (('O',Lb.O),('E0',Lb.E0),('P',Lb.P)): print('EN vs',nm,'rho_between,rho_within,var_b,var_w',dec2(Lb.EN,o))
