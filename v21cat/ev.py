"""ev.py TAG [col]  -> per cut C23/C25/C26: standalone AUC, rho vs M & A, blend gains. TAG e.g. cat_numte  (mean over seeds of ens/{TAG}_c{cut}_s*.npy, col=-1 default)"""
import sys,glob,numpy as np; sys.path.insert(0,'/workspace/huawei/v21seq'); sys.path.insert(0,'/workspace/huawei/v21dcn')
from evalseq import *
from scipy.stats import rankdata
nr=lambda x:(rankdata(x)-1)/(len(x)-1)
H='/workspace/huawei'
def Mfun(c): return np.mean([tologit(np.load(f)[:,-1]) for f in sorted(glob.glob(f'{H}/v21seq/ens/ctl6_A_c{c}_s*.npy'))],0)
def Afun(c):
    fs=sorted(glob.glob(f'{H}/v21seq/ens/ctl6_A_c{c}_s*.npy'))+sorted(glob.glob(f'{H}/v21seq/ens/v21seq_big_A_c{c}_s*.npy')); return np.mean([tologit(np.load(f)[:,-1]) for f in fs],0)
def Dfun(c):
    if c==25: L=[('v21fit/ens/dcn4_A_c25_s0',2),('v21fit/ens/dcn6_A_c25_s0',2),('v21fit/ens/dcn7_A_c25_s0',1)]
    elif c==23: L=[('v21dcn/ens/r1_cr64_A_c23_s0',1)]
    else: return None
    return np.mean([tologit(np.load(f'{H}/{f}.npy')[:,k]) for f,k in L],0)
def cand(tag,c,col=-1):
    fs=sorted(glob.glob(f'{H}/v21cat/ens/{tag}_c{c}_s*.npy')); fs=[f for f in fs if not f.endswith('_gain.json')]
    return np.mean([np.load(f)[:,col] for f in fs],0),len(fs)
if __name__=='__main__':
    tag=sys.argv[1]; col=int(sys.argv[2]) if len(sys.argv)>2 else -1; W=(.05,.1,.15,.2,.3,.5)
    for c in (23,25,26):
        try: C,n=cand(tag,c,col)
        except Exception as e: continue
        if n==0: continue
        y=yv(c); Cr=nr(C); M=nr(Mfun(c)); A=nr(Afun(c)); out=f'C{c} (n{n}) auc {AUC(y,C):.5f} rho_M {np.corrcoef(Cr,M)[0,1]:.4f} rho_A {np.corrcoef(Cr,A)[0,1]:.4f}'
        if c!=26:
            z=nr(zbase('A',c)); b1=.4*z+.6*M; base=AUC(y,b1); g={w:AUC(y,(1-w)*b1+w*Cr)-base for w in W}
            out+=f' | base {base:.5f} gain(.4z+.6M) '+' '.join(f'w{w}:{g[w]*1e5:+.0f}' for w in W)
            D=Dfun(c)
            if D is not None:
                Dr=nr(D); b2=.3*z+.3*M+.4*Dr; base2=AUC(y,b2); g2={w:AUC(y,(1-w)*b2+w*Cr)-base2 for w in W}
                out+=f' | stack(+DCN) {base2:.5f} gain '+' '.join(f'w{w}:{g2[w]*1e5:+.0f}' for w in W)
        print(out)
