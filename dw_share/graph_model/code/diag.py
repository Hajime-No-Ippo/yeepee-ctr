"""diag.py TAG STEP : graph feats on val23 (k=3) / val25 (k=5) / val26 (k=6 if emb exists); standalone, blend gain, and cross-fit (by user, 2-fold) LGBM on [M,z logits + graph feats]."""
import sys,numpy as np; sys.dont_write_bytecode=True
from evalg import *; from gfeat import *
tag,step=sys.argv[1],sys.argv[2]
KOF={23:3,25:5,26:6}
TAGC={23:'u1b',25:'u1',26:'u1'}; STEP={23:'800',25:'350',26:'300'}
def getG(c):
    e=np.load(O+f'emb_{TAGC[c]}_k{KOF[c]}_s0.npz'); return feats(CUTS[c],e[f'U{STEP[c]}'],e[f'V{STEP[c]}'])
for c in (23,25):
    try: F=getG(c)
    except Exception as ex: print('skip',c,ex); continue
    for nme in ('s_uv','s_h','cos_max','cos_mean'): report(c,nme,F[nme])
    comb=demean(F['s_uv'])+demean(F['s_h']); report(c,'dm(uv)+dm(h)',comb)

# ---- cross-fit diagnostic (2-fold by user within val): [M,z] vs [M,z]+graph feats
import lightgbm as lgb
def crossfit(c,X,y,usr,seed=0):
    ug=np.unique(usr); rng=np.random.RandomState(seed); fl=rng.randint(0,2,len(ug)); f=fl[np.searchsorted(ug,usr)]; p=np.zeros(len(y))
    for k in (0,1):
        tr=f!=k; m=lgb.LGBMClassifier(n_estimators=150,learning_rate=.05,num_leaves=15,min_child_samples=200,subsample=.8,subsample_freq=1,colsample_bytree=.8,reg_lambda=10,n_jobs=2,verbose=-1)
        m.fit(X[tr],y[tr]); p[~tr]=m.predict_proba(X[~tr])[:,1]
    return p
for c in (23,25):
    try: F=getG(c)
    except Exception: continue
    y,z,M,b=ref(c); T=load_tab(CUTS[c]); usr=T['user']
    cd={k:demean(v) for k,v in F.items() if k in('s_uv','s_h','cos_max','cos_mean')}
    X0=np.c_[M,z]; X1=np.c_[M,z,F['s_uv'],F['s_h'],F['cos_max'],F['cos_mean'],cd['s_uv'],cd['s_h'],cd['cos_max'],cd['cos_mean'],F['hlen']]
    p0=crossfit(c,X0,y,usr); p1=crossfit(c,X1,y,usr)
    print(f'C{c} crossfit: [M,z] {AUC(y,p0):.5f} (blend {b:.5f})  +graph {AUC(y,p1):.5f}  delta {(AUC(y,p1)-AUC(y,p0))*1e5:+.0f}e-5',flush=True)
