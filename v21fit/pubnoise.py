import sys,resource; sys.path.insert(0,'/workspace/huawei/v21seq')
import glob
from evalseq import *
from scipy.stats import rankdata
nr=lambda x:(rankdata(x)-1)/(len(x)-1)
c=25;y=yv(c); z=nr(zbase('A',c)); fp=sorted(glob.glob(f'/workspace/huawei/v21seq/ens/ctl6_A_c{c}_s*.npy'))
M=nr(np.mean([tologit(np.load(f)[:,-1]) for f in fp],0))
D=nr(np.mean([tologit(np.load(f'ens/{t}_A_c25_s0.npy')[:,e]) for t,e in (('dcn4',2),('dcn6',2),('dcn7',1))],0))
x=.4*z+.6*M; 
n=len(y)//10; print('slates',n)
from sklearn.metrics import roc_auc_score
rs=np.random.RandomState(0)
for lam in (.1,.25):
    yy=nr(x+lam*(D-x)); print('lam',lam,'rho',round(np.corrcoef(x,yy)[0,1],5),'full diff',round(AUC(y,yy)-AUC(y,x),6))
    for frac in (1.0,.3):
        ds=[]
        for b in range(60):
            s=rs.choice(n,int(n*frac),replace=True); ii=(s[:,None]*10+np.arange(10)).ravel()
            ds.append(AUC(y[ii],yy[ii])-AUC(y[ii],x[ii]))
        print('  frac',frac,'sd of diff %.6f'%np.std(ds))
print('maxrss GB',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6)
