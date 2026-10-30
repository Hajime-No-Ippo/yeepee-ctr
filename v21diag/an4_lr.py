from blend4 import *
from combine3 import get
for cut,which in ((25,'blend'),(23,'dwbig')):
    y,Z=get(cut,'B',which); Ls=tologit(np.mean([np.load(f'ens/v21listB_c{cut}_s{s}.npy')[:,3] for s in (0,1,2)],0)).reshape(-1,10)
    R=np.load(f'ens/v21lrB_c{cut}_s0.npy')[:,2].reshape(-1,10); base=.4*Z+.6*Ls
    print(cut,'LR alone',round(AUC(y,R.ravel()),5),'base(.4 v19+.6 list)',round(AUC(y,base.ravel()),5),'corr LR/base',round(np.corrcoef(R.ravel(),base.ravel())[0,1],4))
    for w in (.05,.1,.2,.3): print('  +LR w',w,round(AUC(y,((1-w)*base+w*R).ravel()),5), ' on v19 only:',round(AUC(y,((1-w)*Z+w*R).ravel()),5))
