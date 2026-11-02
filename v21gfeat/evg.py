"""evg.py TAG_NUMG TAG_CTRL [graph_file_pattern]: standalone/rho/gains for gfeat model vs control, and 3-way blend with graph member beyond base4+graph.
base4 = rank-mean(z,M,big,D) equal weights (Code Buddy eq4) ; graph member = v21graph out/graph_c{c}.npy (7-seed raw); col: C25->900 rounds, C23/C26->600 rounds (as v21cat)."""
import os,sys,glob,numpy as np; sys.dont_write_bytecode=True
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score as AUC
sys.path.insert(0,'/workspace/huawei/v21seq'); from evalseq import tologit,yv,zbase
H='/workspace/huawei'; nr=lambda x:(rankdata(x)-1)/(len(x)-1)
ml=lambda pat:np.mean([tologit(np.load(f)[:,-1]) for f in sorted(glob.glob(pat))],0)
mm=lambda L:np.mean([tologit(np.load(f'{H}/{p}.npy')[:,c]) for p,c in L],0)
def D(c): return mm([('v21fit/ens/dcn4_A_c25_s0',2),('v21fit/ens/dcn6_A_c25_s0',2),('v21fit/ens/dcn7_A_c25_s0',1)]) if c==25 else mm([('v21dcn/ens/r0_base_A_c23_s0',1),('v21dcn/ens/r1_a01_A_c23_s0',1),('v21dcn/ens/r1_a1_A_c23_s0',1),('v21dcn/ens/r1_cr64_A_c23_s0',2)])
COL={23:1,25:2,26:1}
def mem(c,tag): return np.load(f'/workspace/huawei/v21gfeat/ens/{tag}_c{c}_s0.npy')[:,COL[c]].astype(np.float64)
def run(tagF,tagC,gfile=None,cuts=(23,25,26)):
    for c in cuts:
        y=yv(c); M=nr(ml(f'{H}/v21seq/ens/ctl6_A_c{c}_s*.npy')); B=nr(ml(f'{H}/v21seq/ens/v21seq_big_A_c{c}_s*.npy'))
        G=nr(np.load(gfile.format(c=c) if gfile else f'{H}/v21graph/out/graph_c{c}.npy').astype(np.float64)); F=nr(mem(c,tagF)); C=nr(mem(c,tagC))
        if c!=26: z=nr(zbase('A',c)); Dd=nr(D(c)); base=.25*(z+M+B+Dd); b1=.4*z+.6*M; b2=.3*z+.3*M+.4*Dd; bn='eq4'
        else: base=.5*(M+B); b1=M; b2=None; bn='M/big'
        a0=AUC(y,base); print(f'\n=== C{c} base({bn}) {a0:.5f} | graph {AUC(y,G):.5f} | gfeat {AUC(y,F):.5f} (rho_M {np.corrcoef(F,M)[0,1]:.4f}, rho_G {np.corrcoef(F,G)[0,1]:.4f}) | control {AUC(y,C):.5f} (rho_M {np.corrcoef(C,M)[0,1]:.4f})',flush=True)
        for nm,X in (('gfeat',F),('control',C),('graph',G)):
            s=f'  {nm:8s} gain over {bn}: '+' '.join(f'w{w}:{(AUC(y,(1-w)*base+w*X)-a0)*1e5:+.0f}' for w in (.1,.15,.2,.3))
            for k,bb in (('.4z+.6M',b1),('.3z+.3M+.4D',b2)):
                if bb is not None: a=AUC(y,bb); s+=f' | {k}: '+' '.join(f'w{w}:{(AUC(y,(1-w)*bb+w*X)-a)*1e5:+.0f}' for w in (.1,.15,.2,.3))
            print(s,flush=True)
        # base+graph best w
        bg={w:AUC(y,(1-w)*base+w*G) for w in (.05,.075,.1,.125,.15,.2)}; wb=max(bg,key=bg.get); ab=bg[wb]; print(f'  base+graph best w={wb}: {ab:.5f} ({(ab-a0)*1e5:+.0f}e-5)  all: '+' '.join(f'{w}:{(v-a0)*1e5:+.0f}' for w,v in bg.items()),flush=True)
        for nm,X in (('gfeat',F),('control',C)):
            res={}
            for w in (0,.05,.1,.15):
                for v in (.05,.1,.15,.2,.3,.4):
                    res[(w,v)]=AUC(y,(1-w-v)*base+w*G+v*X)
            best=max(res,key=res.get); print(f'  3-way base+graph+{nm}: best (wG,v)={best} {res[best]:.5f} => beyond base+graph(best w) {(res[best]-ab)*1e5:+.1f}e-5 | at wG=.1: '+' '.join(f'v{v}:{(res[(.1,v)]-ab)*1e5:+.0f}' for v in (.05,.1,.15,.2,.3,.4)),flush=True)
if __name__=='__main__':
    run(sys.argv[1],sys.argv[2],sys.argv[3] if len(sys.argv)>3 else None)
