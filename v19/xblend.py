import sys,numpy as np
from blend19 import *
tag=sys.argv[1]; col=int(sys.argv[2]); v=sys.argv[3] if len(sys.argv)>3 else 'B'
b,d,ml,z=comp(v,'nm'); x=np.load(f'ens/{tag}_c25_s0.npy')[:,col]
print('standalone %.5f nm blend %.5f'%(AUC(y,x),AUC(y,z)))
sd=lambda a:(a-a.mean())/a.std()
for w in (0,.1,.2,.3,.4,.5): print(w,'%.5f'%AUC(y,(1-w)*sd(z)+w*sd(x)))
