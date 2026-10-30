from blend4 import *
from combine3 import get
Pm25=np.load('ens/pslate_c25_B.npy')[:,0]; Pm25-=Pm25.mean(); P23=np.load('ens/pslate_c23_B.npy')[:,0]; P23-=P23.mean()
res={}
for cut,which,P in ((25,'blend',Pm25),(23,'dwbig',P23)):
    y,Z=get(cut,'B',which)
    for nm,seeds in (('s0',[0]),('3seed',[0,1,2])):
        L=tologit(np.mean([np.load(f'ens/v21listB_c{cut}_s{s}.npy')[:,3] for s in seeds],0)); print(cut,nm,stats(L,y,'list'))
        Ls=L.reshape(-1,10)
        print('  base',round(AUC(y,Z.ravel()),5),'v21slate',round(AUC(y,comb(Z,P,.3,.5).ravel()),5))
        for w in (.4,.5,.6,.7,1.0):
            z=(1-w)*Z+w*Ls
            print('  w',w,round(AUC(y,z.ravel()),5),'+corr(.3,.5)',round(AUC(y,comb(z,P,.3,.5).ravel()),5),'(.1,.25)',round(AUC(y,comb(z,P,.1,.25).ravel()),5),'(.2,.5)',round(AUC(y,comb(z,P,.2,.5).ravel()),5))
    # bce
    if cut==25:
        B=tologit(np.load('ens/v21bceB_c25_s0.npy')[:,3]) if False else np.load('ens/v21bceB_c25_s0.npy')[:,3]
        print('bce ep3 standalone',stats(B,y,'bce'))
        for w in (.3,.5): print('  bce w',w,round(AUC(y,((1-w)*Z+w*B.reshape(-1,10)).ravel()),5))
