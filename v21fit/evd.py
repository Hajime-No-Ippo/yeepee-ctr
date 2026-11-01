import sys; sys.path.insert(0,'/workspace/huawei/v21seq')
from evalseq import *
import glob
from scipy.special import expit
from scipy.stats import spearmanr
c=int(sys.argv[1]); y=yv(c); z=zbase('A',c); L,_=listp('A',c); T=.4*z+.6*L
fs=sorted(glob.glob(f'/workspace/huawei/v21fit/ens/dsim_A_c{c}_s*.npy')); S=np.mean([tologit(np.load(f)[:,-1]) for f in fs],0)
fp=sorted(glob.glob(f'/workspace/huawei/v21seq/ens/ctl6_A_c{c}_s*.npy'))[:len(fs)]; M=np.mean([tologit(np.load(f)[:,-1]) for f in fp],0)
print('n seeds',len(fs),'teacher',AUC(y,T),'student',AUC(y,S),'plain ctl6(same seeds)',AUC(y,M))
print('rho(S,T)',spearmanr(S[:200000],T[:200000])[0],'rho(M,T)',spearmanr(M[:200000],T[:200000])[0])
for w in (.1,.2,.3,.5,.7):
    print(w,'T+S',round(AUC(y,(1-w)*T+w*S)-AUC(y,T),5),'T+M',round(AUC(y,(1-w)*T+w*M)-AUC(y,T),5),'.4z+.6M',round(AUC(y,.4*z+.6*M)-AUC(y,T),5),'.4z+.6S',round(AUC(y,.4*z+.6*S)-AUC(y,T),5))
