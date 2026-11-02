"""assemble.py : multi-seed graph feats per cut -> out/gfeat_c{c}.npz, graph_simple / graph_pure / graph_head fold preds + report. python assemble.py [cuts]"""
import sys,os,glob,numpy as np,lightgbm as lgb; sys.dont_write_bytecode=True
from evalg import *; from gfeat import *
KOF={23:3,25:5,26:6}; TAGC={23:'u1b',25:'u1',26:'u1'}; STEP={23:'800',25:'350',26:'350'}
cuts=[int(x) for x in sys.argv[1:]] or [23,25,26]
NAMES=('s_uv','s_h','cos_max','cos_mean')
def multiseed(c):
    fs=sorted(glob.glob(O+f'emb_{TAGC[c]}_k{KOF[c]}_s*.npz')); acc=None; n=0
    for f in fs:
        e=np.load(f)
        if f'U{STEP[c]}' not in e.files: continue
        F=feats(CUTS[c],e[f'U{STEP[c]}'],e[f'V{STEP[c]}']); n+=1
        if acc is None: acc={k:F[k].astype(np.float64) for k in F}
        else:
            for k in NAMES: acc[k]+=F[k]
    for k in NAMES: acc[k]/=n
    return acc,n
def crossfit(X,y,usr,seed=0,nest=200):
    ug=np.unique(usr); fl=np.random.RandomState(seed).randint(0,2,len(ug)); f=fl[np.searchsorted(ug,usr)]; p=np.zeros(len(y))
    for k in (0,1):
        tr=f!=k; m=lgb.LGBMClassifier(n_estimators=nest,learning_rate=.05,num_leaves=15,min_child_samples=200,subsample=.8,subsample_freq=1,colsample_bytree=.8,reg_lambda=10,n_jobs=2,verbose=-1)
        m.fit(X[tr],y[tr]); p[~tr]=m.predict_proba(X[~tr])[:,1]
    return p
def gfmat(F,P=None):
    cols=[F[k] for k in NAMES]+[demean(F[k]) for k in NAMES]+[F['hlen']]
    if P is not None: cols+= [P[:,i] for i in range(P.shape[1])]+[demean(P[:,i]) for i in range(P.shape[1])]
    return np.c_[tuple(cols)]
SUM=[]
for c in cuts:
    F,n=multiseed(c); T=load_tab(CUTS[c]); usr=T['user']; y,z,M,b=ref(c)
    pp=O+f'ppr_{CUTS[c]}.npy'; P=np.load(pp) if os.path.exists(pp) else None
    G0=demean(F['s_uv'])+demean(F['s_h'])+demean(F['cos_mean'])
    np.savez(O+f'gfeat_c{c}.npz',**{k:F[k] for k in NAMES},hlen=F['hlen'])
    X=gfmat(F,P); G1=crossfit(X,y,usr); G2=crossfit(np.c_[z,X],y,usr)
    zl=np.c_[z]; Z0=crossfit(zl,y,usr)
    np.save(O+f'graph_simple_c{c}.npy',G0.astype(np.float32)); np.save(O+f'graph_pure_c{c}.npy',G1.astype(np.float32)); np.save(O+f'graph_head_c{c}.npy',G2.astype(np.float32))
    print(f'== C{c} seeds {n} ppr {P is not None}  z-only crossfit auc {AUC(y,Z0):.5f}  z raw {AUC(y,z):.5f}',flush=True)
    for nm,G in (('simple',G0),('pure',G1),('head(z+graph)',G2)): report(c,nm,G)
    # stack test: [M,z]+graph crossfit
    p0=crossfit(np.c_[M,z],y,usr); p1=crossfit(np.c_[M,z,X],y,usr); print(f'C{c} stack crossfit [M,z] {AUC(y,p0):.5f} +graph {AUC(y,p1):.5f} delta {(AUC(y,p1)-AUC(y,p0))*1e5:+.0f}e-5 (blend .4z+.6M {b:.5f})',flush=True)
