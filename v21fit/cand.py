import sys; sys.path.insert(0,'/workspace/huawei/v21seq')
import glob
from evalseq import *
from scipy.stats import rankdata
nr=lambda x:(rankdata(x)-1)/(len(x)-1)
c=int(sys.argv[1]); tags=sys.argv[2:]
y=yv(c); z=nr(zbase('A',c)); fp=sorted(glob.glob(f'/workspace/huawei/v21seq/ens/ctl6_A_c{c}_s*.npy'))
M=nr(np.mean([tologit(np.load(f)[:,-1]) for f in fp],0)); M0=nr(tologit(np.load(fp[0])[:,-1]))
base=.4*z+.6*M; print('C',c,'z',round(AUC(y,z),5),'M8',round(AUC(y,M),5),'M s0',round(AUC(y,M0),5),'base .4z+.6M8',round(AUC(y,base),5),'base s0',round(AUC(y,.4*z+.6*M0),5))
for t in tags:
    f=f'ens/{t}_A_c{c}_s0.npy'; X=np.load(f); 
    for e in ((int(__import__("os").environ["EP"]),) if "EP" in __import__("os").environ else (X.shape[1]-1,)):
        q=nr(tologit(X[:,e])); a=AUC(y,q); rho=np.corrcoef(q,M0)[0,1]; rz=np.corrcoef(q,z)[0,1]
        print(t,'ep',e,'AUC',round(a,5),'rho(M0)',round(rho,4),'rho(z)',round(rz,4),'| +cand into .4z+.6M8 at w.2/.35/.5:',[round(AUC(y,(1-w)*base+w*q),5) for w in (.2,.35,.5)],'| M0->cand swap .4z+.6q',round(AUC(y,.4*z+.6*q),5),'| .4z+.3M0+.3q',round(AUC(y,.4*z+.3*M0+.3*q),5),'| .4z+.6M0',round(AUC(y,.4*z+.6*M0),5))
