import sys,numpy as np; sys.dont_write_bytecode=True
sys.path.insert(0,'/workspace/huawei/v21graph')
from evalg import *
for c in (23,25):
    y,z,M,b=ref(c); A=.4*z+.6*M; D=nr(load('r0_base',c)[0])
    G7=nr(np.load(f'/workspace/huawei/v21graph/out/graph_c{c}.npy').astype(np.float64)); G8=nr(np.load(f'/workspace/huawei/v21graph2/out/graph2s3b_f6_c{c}.npy').astype(np.float64))
    L=nr(.335*A+.665*D); a0=AUC(y,L)
    print('C%d: base L(q=.665) %.5f | G7 %.5f G8 %.5f rho(G7,G8) %.3f'%(c,a0,AUC(y,G7),AUC(y,G8),np.corrcoef(G7,G8)[0,1]))
    print(' DIR1 gain e-5 vs L alone; rows share7 (0,.5,1), cols total graph weight (L+w*G, rel):')
    ws=(.1,.15,.2,.25,.3)
    for s in (0,.25,.5,.75,1): print('  s7=%.2f '%s+' '.join('%+6.1f'%((AUC(y,L+w*(s*G7+(1-s)*G8))-a0)*1e5) for w in ws))
    print(' DIR2: scale of ours/D base vs graph (same as total w above, i.e. lowering base share = raising w): see rows above; best w locally ~.2-.3 (rel to L std units)')
    print(' DIR3: q (D share, ->Q2-like at 1) with G8 at w .2 and G7+G8 mix: ')
    for q in (.33,.5,.665,.8,1): 
        Lq=nr((1-q)*A+q*D); print('  q=%.2f '%q+'G8 %+6.1f  mix %+6.1f'%((AUC(y,Lq+.2*G8)-AUC(y,Lq))*1e5,(AUC(y,Lq+.2*(.5*G7+.5*G8))-AUC(y,Lq))*1e5)+'   abs AUC(Lq+.2G8) %.5f'%AUC(y,Lq+.2*G8))
