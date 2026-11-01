"""evalseq.py VAR CUT TAG [col] [seeds]: standalone + blend AUCs vs v21list (same var).  base z = v19 blend (C25) / dwbig (C23)."""
import sys, numpy as np, polars as pl
from sklearn.metrics import roc_auc_score as AUC
H='/workspace/huawei'
def tologit(lp): lp=np.minimum(lp,-1e-6); return lp-np.log1p(-np.exp(lp))
def yv(c): return pl.read_parquet(f'{H}/v12/feat_v3/val{c}.parquet',columns=['clicked'])['clicked'].to_numpy()
def zbase(v,c):
    L=lambda f,col:np.load(f'{H}/v19/ens/{f}.npy')[:,col]; b=L(f'nm{v}s_c{c}_s0',0)
    if c==23: return b
    d=L(f'nm{v}s_s_c{c}_s0',1); ml=np.mean([L(f'nm{v}smlp_c{c}_s{k}',3) for k in (0,1,2)],0); return (b+d+2*ml)/4
def listp(v,c):
    import glob; fs=sorted(glob.glob(f'{H}/v21diag/ens/v21list{v}_c{c}_s*.npy')); return tologit(np.mean([np.load(f)[:,3] for f in fs],0)),len(fs)
def seqp(path_glob,col=-1):
    import glob; fs=sorted(glob.glob(path_glob)); return tologit(np.mean([np.load(f)[:,col] for f in fs],0)),len(fs)
if __name__=='__main__':
    v,c,pat=sys.argv[1],int(sys.argv[2]),sys.argv[3]; col=int(sys.argv[4]) if len(sys.argv)>4 else -1
    y=yv(c); z=zbase(v,c); Lg,nl=listp(v,c); S,ns=seqp(pat,col)
    print(f'{v} C{c}: z(v19/dwbig) {AUC(y,z):.5f} | list({nl} seeds) {AUC(y,Lg):.5f} | seq({ns} seeds) {AUC(y,S):.5f}')
    print(' list-blend(.4z+.6L) %.5f'%AUC(y,.4*z+.6*Lg))
    for ws in (.2,.3,.4,.5,.6,.8,1.0):
        print(f' w_seq={ws}: seq+z (1-w)z+wS {AUC(y,(1-ws)*z+ws*S):.5f} | list-blend swap: .4z+.6[(1-w)L+wS] {AUC(y,.4*z+.6*((1-ws)*Lg+ws*S)):.5f}')
