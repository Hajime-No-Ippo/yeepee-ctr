import sys; sys.path.insert(0,'/workspace/huawei/v21graph'); import numpy as np
from gfeat import *
for nm,tag,k,st in [('f3','u1b',3,800),('val23','u1b',3,800),('f5','u1',5,350),('val25','u1',5,350),('f6','u1',6,350),('val26','u1',6,350),('eval','u1',7,350)]:
    e=np.load(O+f'emb_{tag}_k{k}_s0.npz'); F=feats(nm,e[f'U{st}'],e[f'V{st}'])
    print(nm,{n:(round(float(F[n].mean()),3),round(float(F[n].std()),3)) for n in ('s_uv','s_h','cos_mean','cos_max')},flush=True)
