import sys,numpy as np; sys.dont_write_bytecode=True
sys.path.insert(0,'/workspace/huawei/v21graph')
from evalg import *
O='/workspace/huawei/v21graph2/out/'
for c in (23,25,26):
    y,z,M,b=ref(c); A=.4*z+.6*M
    try: D=nr(load('r0_base',c)[0])
    except Exception as e: print(c,'no D',e); D=None
    G=nr(np.load(O+f'graph2s3b_f6_c{c}.npy').astype(np.float64)); a0=AUC(y,A)
    print('C%d: A(.4z+.6M) %.5f D %s G8 %.5f'%(c,a0,'%.5f'%AUC(y,D) if D is not None else '-',AUC(y,G)))
    if D is None: continue
    ws=(0,.05,.1,.15,.2,.3)
    print(' rows=q (D share), cols=graph weight (rel) ', ws, ' [AUC-AUC(A)] e-5')
    for q in (0,.33,.5,.67,.8,1):
        L=nr((1-q)*A+q*D); print(' q=%.2f '%q+' '.join('%+6.1f'%((AUC(y,(1-w)*L+w*G)-a0)*1e5) for w in ws))
