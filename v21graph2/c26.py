import sys,glob,numpy as np; sys.dont_write_bytecode=True
from ana import *
sys.path.insert(0,'/workspace/huawei/v21seq')
from evalseq import tologit,yv
c=26; y=yv(26)
fp=sorted(glob.glob('/workspace/huawei/v21seq/ens/ctl6_A_c26_s*.npy')); M=nr(np.mean([tologit(np.load(f)[:,-1]) for f in fp],0)); a0=AUC(y,M)
def run(n,tag='u1',step=350):
    A=avg(tag,c,SEEDORD[:n],step); Z=np.stack([zs(A[:,j]) for j in range(4)],1)
    for wts in ((1,1,1,0),(2,0,1,0),(2,.5,.5,0)):
        G=nr(Z@np.array(wts,float)); print(n,wts,'auc %.5f'%AUC(y,G),' '.join('w%.3f %+.1f'%(w,(AUC(y,(1-w)*M+w*G)-a0)*1e5) for w in (.1,.125,.15,.2)),flush=True)
run(7)
