import sys; sys.path.insert(0,'/workspace/huawei/dw_levers')
import numpy as np, time
from dwcommon import *
t=time.time(); D=Data(); print('load',time.time()-t, D.NE, D.NU, D.NI, D.Hlen.max(), D.Hlen.mean(), D.ETS.min()-J20, D.ETS.max()-J20, flush=True)
print(np.searchsorted(D.ETS,Jd(25)), np.searchsorted(D.ETS,Jd(23)))
v=D.val(25); print(len(v['y']), v['ts'].min()-Jd(25), v['ts'].max()-Jd(25))
# is history ordered oldest first?
e=100000; print(D.HT[D.OFF[e]:D.OFF[e+1]]-D.ETS[e])
import resource; print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6,'GB')
