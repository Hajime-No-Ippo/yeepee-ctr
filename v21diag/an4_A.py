from blend4 import *
from combine3 import get
for cut,which in ((25,'blend'),(23,'dwbig')):
    y,Z=get(cut,'A',which); L=tologit(np.load(f'ens/v21listA_c{cut}_s0.npy')[:,3]).reshape(-1,10)
    print(cut,'A list s0',stats(L.ravel(),y,'l'),'base',round(AUC(y,Z.ravel()),5))
    print('  ',[(w,round(AUC(y,((1-w)*Z+w*L).ravel()),5)) for w in (.4,.5,.6,.7)])
