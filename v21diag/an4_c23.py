from blend4 import *
from combine3 import get
y3,Z3=get(23,'B','dwbig'); L3=tologit(np.load('ens/v21listB_c23_s0.npy')[:,3]).reshape(-1,10)
P3=np.load('ens/pslate_c23_B.npy')[:,0]; P3=P3-P3.mean()
print(stats(Z3.ravel(),y3,'C23 dwbig')); print(stats(L3.ravel(),y3,'C23 list'))
print('C23 v21slate on dwbig',round(AUC(y3,comb(Z3,P3,.3,.5).ravel()),5))
for w in (0,.25,.5,.75,1):
    z=(1-w)*Z3+w*L3; print('C23 w',w,round(AUC(y3,z.ravel()),5),'corr(.3,.5)',round(AUC(y3,comb(z,P3,.3,.5).ravel()),5),'corr(.1,.25)',round(AUC(y3,comb(z,P3,.1,.25).ravel()),5))
