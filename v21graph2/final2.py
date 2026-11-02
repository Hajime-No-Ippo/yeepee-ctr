"""final2.py seeds(comma) name : L8 u12 650 steps, variants sum3 / sum3b. saves out/graph2{v}_{name}_c{23,25,26}.npy (float32, v21seq val order), eval arrays, prints table"""
import sys,glob,numpy as np; sys.dont_write_bytecode=True
from evalrho import *; from ana import *; import ana
sys.path.insert(0,'/workspace/huawei/v21seq')
from evalseq import tologit,yv
seeds=[int(s) for s in sys.argv[1].split(',')]; name=sys.argv[2]; WSX=(.05,.1,.15,.2,.25,.3)
V={'s3':(1,1,1,0),'s3b':(2,0,1,0)}
y26=yv(26); fp=sorted(glob.glob('/workspace/huawei/v21seq/ens/ctl6_A_c26_s*.npy')); M26=nr(np.mean([tologit(np.load(f)[:,-1]) for f in fp],0)); a26=AUC(y26,M26)
Z={c:(lambda A:np.stack([zs(A[:,j]) for j in range(4)],1))(avg('u12',c,seeds,650)) for c in (23,25,26,27)}
print(name,'seeds',seeds)
for v,w in V.items():
    w=np.array(w,float)
    for c in (23,25,26):
        G=Z[c]@w; np.save(O+f'graph2{v}_{name}_c{c}.npy',G.astype(np.float32))
        if c==26:
            Gn=nr(G); print(f'C26 {v} auc {AUC(y26,G):.5f} rhoM {np.corrcoef(Gn,M26)[0,1]:.3f} vsM '+' '.join('w%.2f %+.1f'%(x,(AUC(y26,(1-x)*M26+x*Gn)-a26)*1e5) for x in WSX),flush=True)
        else:
            a,r,o=best(c,G,WSX); print(f'C{c} {v} auc {a:.5f} rhoM {r:.3f} '+' | '.join(f'{k}: best {q[0]:+.1f}@{q[1]} {q[2]}' for k,q in o.items()),flush=True)
    G=Z[27]@w; np.save(O+f'graph2{v}_{name}_eval_v21seqorder.npy',G.astype(np.float32)); gm=to_merged(G); np.save(O+f'graph2{v}_{name}_eval_merged_rank.npy',nr(gm))
    print(f'eval {v}: rho EN {rho(gm):.4f} rho G7 {rho(gm,nr(G7)):.4f}',flush=True)
np.save(O+'graph2_eval_rowid.npy',np.asarray(rid))
