import sys,time,resource; t=time.time()
sys.dont_write_bytecode=True
sys.path.insert(0,'/workspace/huawei/dw_signals')
from common import *
D=Data(); print(D.NE,D.NU,D.NI,len(D.HI),time.time()-t,flush=True)
import numpy as np
print('clicks/event',D.Y.sum()/D.NE,'hist len mean',D.Hlen.mean())
for k in range(1,8):
    S=D.snap(J20+k*DAY); print(k,S['nX'],len(S['cu']),len(np.unique(S['cu'])),len(np.unique(S['ci'])),flush=True)
ce,cj=np.nonzero(D.Y); day=((D.ETS[ce]-J20)//DAY).astype(int)
print(np.bincount(day-day.min()),day.min())
print('maxrss GB',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6)
