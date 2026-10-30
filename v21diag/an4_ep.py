from blend4 import *
from combine3 import get
for cut,v,which in ((25,'B','blend'),(23,'B','dwbig')):
    y,Z=get(cut,v,which); E=np.load(f'ens/v21listB_c{cut}_s0.npy')
    for nm,L in (('ep3',tologit(E[:,3])),('ep2+3',tologit((E[:,2]+E[:,3])/2))):
        L=L.reshape(-1,10); print(cut,nm,[round(AUC(y,((1-w)*Z+w*L).ravel()),5) for w in (0,.4,.5,.6)])
