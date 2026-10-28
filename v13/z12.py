import numpy as np
def z12(c):
    s=lambda t,k:np.load(f'/workspace/huawei/v12/ens/{t}_c{c}_s0.npy')[:,k]
    m=np.mean([np.load(f'/workspace/huawei/v12/ens/dwmlp_c{c}_s{k}.npy')[:,3] for k in (range(3) if c!=26 else [0])],0)
    return ((s('dwbig',1)+s('dw_s',1)+2*m)/4).astype(np.float64)
