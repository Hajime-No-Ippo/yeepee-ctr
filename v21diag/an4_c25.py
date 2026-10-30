from blend4 import *
y,Zb=get(25,'B','blend') if False else (None,None)
from combine3 import get
y,Zb=get(25,'B','blend'); y,Zd=get(25,'B','dwbig')
L=tologit(lst('v21listB',25,[0],3)); Ls=[tologit(np.load('ens/v21listB_c25_s0.npy')[:,e]) for e in range(4)]
print(stats(Zb.ravel(),y,'v19 blend B')); print(stats(L,y,'list MLP ep3'))
Pm=np.load('ens/pslate_c25_B.npy')[:,0]; Pm=Pm-Pm.mean()
print(stats(comb(Zb,Pm,.3,.5),y,'v21slate B'))
Lz=L.reshape(-1,10)
for w in (0.25,0.5,0.75):
    z=(1-w)*Zb+w*Lz; print(w,stats(z.ravel(),y,f'blend w={w}'))
    for a,c in ((.3,.5),(.2,.5),(.3,.25),(0.1,0.25),(0,0)):
        print('   corr',a,c,round(AUC(y,comb(z,Pm,a,c).ravel()),5))
