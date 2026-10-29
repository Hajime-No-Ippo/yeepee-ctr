import sys; sys.dont_write_bytecode=True
sys.path.insert(0,'/workspace/huawei/dw_v12sig'); sys.path.insert(0,'/workspace/huawei/dw_signals')
from sigs import *
D=Data(); print(D.NE,D.NU,D.NI,len(D.HI))
V=D.val(25); print({k:v.shape for k,v in V.items()}, V['y'].mean())
S=D.snap(Jd(25)); print(len(S['cu']))
import resource; print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6,'GB')
