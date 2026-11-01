import sys,numpy as np
from evalseq import *
v,c,f=sys.argv[1],int(sys.argv[2]),sys.argv[3]; col=int(sys.argv[4]); y=yv(c); z=zbase(v,c); Lg,_=listp(v,c); base=.4*z+.6*Lg
S=np.load(f)[:,col].reshape(-1,10); Sd=(S-S.mean(1,keepdims=True)).reshape(-1)
print('base',round(AUC(y,base),5),'gbdt raw',round(AUC(y,S.reshape(-1)),5))
for w in (.1,.2,.3,.5,.8,1.2): print(' w',w,'base+w*demeaned', round(AUC(y,base+w*Sd),5),' base+w*raw',round(AUC(y,base+w*S.reshape(-1)),5))
