import glob,numpy as np
from evalseq import *
for c in (23,25):
    y=yv(c); z=zbase('A',c); L=tologit(np.load(f'{H}/v21diag/ens/v21listA_c{c}_s0.npy')[:,3])
    fs=sorted(glob.glob(f'ens/ctl6_A_c{c}_s*.npy'))+sorted(glob.glob(f'ens/v21seq_big_A_c{c}_s*.npy')); M=np.mean([tologit(np.load(f)[:,-1]) for f in fs],0)
    print(c,len(fs),'models; base',round(AUC(y,.4*z+.6*L),5))
    for w in (.4,.5,.6,.7,.8,.9,1.0): print('  w',w,'M-only',round(AUC(y,(1-w)*z+w*M),5),' with .5L',round(AUC(y,(1-w)*z+w*(.5*L+.5*M)),5))
