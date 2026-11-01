"""rep.py [tag-glob-substring ...]: per cut standalone AUC per epoch (final), rho with M, blend gains.
base=.4z+.6M(ctl6 8 seeds, rank); new=.3z+.3M+.4D ; best over wd grid"""
import sys,glob,os,json,numpy as np
sys.path.insert(0,'/workspace/huawei/v21seq')
from evalseq import *
from scipy.stats import rankdata
nr=lambda x:(rankdata(x)-1)/(len(x)-1)
CUTS=(23,25); CACHE={}
def ref(c):
    if c in CACHE: return CACHE[c]
    y=yv(c); z=nr(zbase('A',c)); fp=sorted(glob.glob(f'/workspace/huawei/v21seq/ens/ctl6_A_c{c}_s*.npy'))
    M=nr(np.mean([tologit(np.load(f)[:,-1]) for f in fp],0)); base=AUC(y,.4*z+.6*M); CACHE[c]=(y,z,M,base); return CACHE[c]
def evalD(D,c):
    y,z,M,base=ref(c); Dr=nr(D); r=dict(auc=AUC(y,Dr),rho=np.corrcoef(Dr,M)[0,1])
    g={wd:AUC(y,.3*z+(.7-wd)*M+wd*Dr)-base for wd in (.2,.3,.4,.5)}; r['g']=g; r['gb']=max(g.values()); r['g4']=g[.4]; return r
def load(tag,c,seeds=None,ep=-1):
    fs=sorted(glob.glob(f'/workspace/huawei/v21dcn/ens/{tag}_A_c{c}_s*.npy')) if seeds is None else [f'/workspace/huawei/v21dcn/ens/{tag}_A_c{c}_s{s}.npy' for s in seeds]
    return np.mean([tologit(np.load(f)[:,ep]) for f in fs],0),len(fs)
if __name__=='__main__':
    tags=sorted({os.path.basename(f).split('_A_c')[0] for f in glob.glob('/workspace/huawei/v21dcn/ens/*.npy')})
    if len(sys.argv)>1: tags=[t for t in tags if any(s in t for s in sys.argv[1:])]
    print('base: '+' '.join(f'C{c} {ref(c)[3]:.5f}' for c in CUTS))
    for t in tags:
        out=[]
        for c in CUTS:
            if not glob.glob(f'/workspace/huawei/v21dcn/ens/{t}_A_c{c}_s*.npy'): out.append(f'C{c}: -'); continue
            D,n=load(t,c); r=evalD(D,c); eps=' '.join(f'{AUC(ref(c)[0],tologit(np.load(glob.glob(f"/workspace/huawei/v21dcn/ens/{t}_A_c{c}_s*.npy")[0])[:,e])):.5f}'[2:] for e in range(np.load(glob.glob(f"/workspace/huawei/v21dcn/ens/{t}_A_c{c}_s*.npy")[0]).shape[1])); out.append(f'[{eps}] '+f"C{c}(n{n}): auc {r['auc']:.5f} rho {r['rho']:.4f} gain.4 {r['g4']*1e5:+.0f} best {r['gb']*1e5:+.0f} (x1e-5)")
        print(f'{t:14s} '+' | '.join(out))
