#!/usr/bin/env python
"""blend_oof.py -- out-of-fold stack / weight search over 'our own' member predictions (Huawei CTR, variant A only).
Run:  taskset -c 0-3 /workspace/huawei/v20/.venv/bin/python blend_oof.py [--rebuild] [--members z,M,big,D,ref] [--no-report]
ADD A MEMBER = one line in MEMBERS below: name -> {cut: loader()} where loader returns a raw score vector (any monotone scale, e.g. logit)
in val{cut} row order (= /workspace/huawei/v12/feat_v3/val{cut}.parquet). Use mean_logit(glob, col) for seed averages.
Only global rank/N is ever used for blending (no per-user / per-group normalisation). Never writes any eval/submission file.
CPU only, 4 threads (set before numpy import)."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ.setdefault(k,'4')
import sys, glob, json, re, time, itertools
import numpy as np, polars as pl
from scipy.stats import rankdata, spearmanr
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
H='/workspace/huawei'; OUT=f'{H}/cb_blend'; CUTS=(23,25,26)
def AUC(y,s): return roc_auc_score(y,s)
def rk(x): return ((rankdata(x)-1)/(len(x)-1)).astype(np.float64)            # global rank/N
def tologit(lp): lp=np.minimum(np.asarray(lp,np.float64),-1e-6); return lp-np.log1p(-np.exp(lp))
def mean_logit(pattern, col=-1, from_logprob=True):
    fs=sorted(glob.glob(pattern)); assert fs, pattern
    P=[np.load(f)[:,col].astype(np.float64) for f in fs]; P=[tologit(p) for p in P] if from_logprob else P
    return np.mean(P,0), len(fs)
def mean_logit_multi(items):          # items: [(glob_without_ext, col)] each seed/cfg averaged as logit
    return np.mean([tologit(np.load(f'{H}/{p}.npy')[:,c]) for p,c in items],0)
# ---------------- MEMBERS (one line per member; fold -> loader) ----------------
def z_(c):   # v19 stack at C25 (b+d+2*mlp)/4, dwbig at C23 (as v21seq/evalseq.zbase, variant A)
    L=lambda f,col:np.load(f'{H}/v19/ens/{f}.npy')[:,col].astype(np.float64); b=L(f'nmAs_c{c}_s0',0)
    if c==23: return b
    return (b+L(f'nmAs_s_c{c}_s0',1)+2*np.mean([L(f'nmAsmlp_c{c}_s{k}',3) for k in (0,1,2)],0))/4
MEMBERS={
 'z':   {23:lambda:z_(23), 25:lambda:z_(25)},
 'M':   {c:(lambda c=c:mean_logit(f'{H}/v21seq/ens/ctl6_A_c{c}_s*.npy')[0]) for c in (23,25,26)},          # ctl6 8 seeds
 'big': {c:(lambda c=c:mean_logit(f'{H}/v21seq/ens/v21seq_big_A_c{c}_s*.npy')[0]) for c in (23,25,26)},   # v21seq_big 3/3/2 seeds
 'list1':{c:(lambda c=c:tologit(np.load(f'{H}/v21diag/ens/v21listA_c{c}_s0.npy')[:,3])) for c in (23,25)}, # v21list single seed
 # D (DCN-v2 bag). C25 = D3 snapshots as v21fit/mkD (dcn4 ep2, dcn6 ep2, dcn7 ep1). C23: NO matching bag exists -> PROXY from v21dcn r1 configs at their peak epoch
 'D':   {25:lambda:mean_logit_multi([('v21fit/ens/dcn4_A_c25_s0',2),('v21fit/ens/dcn6_A_c25_s0',2),('v21fit/ens/dcn7_A_c25_s0',1)]),
         23:lambda:mean_logit_multi([('v21dcn/ens/r0_base_A_c23_s0',1),('v21dcn/ens/r1_a01_A_c23_s0',1),('v21dcn/ens/r1_a1_A_c23_s0',1),('v21dcn/ens/r1_cr64_A_c23_s0',2)])},
 'ref': {c:(lambda c=c:np.mean([np.load(f'{H}/v20/ens/ref484_c{c}_s{s}.npy').astype(np.float64) for s in (0,1,2)],0)) for c in (23,25)},   # v20 ref484 LightGBM 3 seeds, raw
}
# composites (derived from other members, never fitted as members unless listed in --members)
def A_all(c):   # ctl6 8 seeds + big seeds, one logit mean (v21seq 'ALL')
    P=[tologit(np.load(f)[:,-1]) for t in ('ctl6','v21seq_big') for f in sorted(glob.glob(f'{H}/v21seq/ens/{t}_A_c{c}_s*.npy'))]; return np.mean(P,0)
# v21cat numte LGB (raw score) + v21graph LightGCN feature blend 'graph_simple' (interim multi-seed, saved 02:57). Loaded from FROZEN copies in cb_blend/members
# (cp -p of v21cat/out/cat_numte_c*.npy and v21graph/out/graph_simple_c*.npy at 03:01 Dublin; v21graph's ms.py may overwrite its file later). Row order verified == val{c}.parquet rowid.
MEMBERS['cat']={c:(lambda c=c:np.load(f'{OUT}/members/cat_numte_c{c}.npy').astype(np.float64)) for c in (23,25,26)}
MEMBERS['graph']={c:(lambda c=c:np.load(f'{OUT}/members/graph_simple_c{c}.npy').astype(np.float64)) for c in (23,25)}
# graphU1 = same 'dm sum3' LightGCN member rebuilt by Code Buddy from v21graph/out/emb_*.npz: more seeds (C23 u1b n5, C25 u1 n7, C26 u1 n5), has C26, rowid-verified vs oof. FROZEN in u1u2/graph_members.npz
MEMBERS['graphU1']={c:(lambda c=c:np.load(f'{OUT}/u1u2/graph_members.npz')[f'G{c}'].astype(np.float64)) for c in (23,25,26)}
# graph0312 = Grok Bot's FINAL 7-seed RAW (non-demeaned) graph score G=z(s_uv)+z(s_h)+z(cos_mean) (seeds 0,1,2,3,4,5,9; u1b k3 C23, u1 k5/k6 C25/C26), FROZEN copy in graph0312/ (cp -p of v21graph/out/graph_c*.npy, 03:12). rowid-verified.
MEMBERS['graph0312']={c:(lambda c=c:np.load(f'{OUT}/graph0312/graph_c{c}.npy').astype(np.float64)) for c in (23,25,26)}
MEMBERS['A']={c:(lambda c=c:A_all(c)) for c in (23,25,26)}
# reported standalone AUCs (Grok Bot STATUS.md / v21fit / Code Buddy) for the alignment check; tol 1e-4 (1.5e-4 for approximate figures)
# ref: reported per-seed mean .79664/.81749 (verified); seed-average of 3 = .79707/.81778
REPORTED={'z':{23:.79679,25:.81829},'M':{23:.79862,25:.81904,26:.82797},'big':{23:.79851,25:.81916,26:.82791},'A':{23:.79864,25:.81910,26:.82798},
          'list1':{23:.79799,25:.81857},'D':{25:.8189},'ref':{23:.79707,25:.81778},
          'cat':{23:.7965,25:.8164,26:.8249},'graph':{23:.765,25:.796},'graphU1':{23:.76844,25:.79835,26:.80783},'graph0312':{23:.76818,25:.79514,26:.80560}}
TOL={'graph':2e-3,'graphU1':2e-4,'graph0312':2e-4}   # graph figures rounded to 3 dp + file may hold fewer seeds than STATUS's interim numbers
def labels(c):
    d=pl.read_parquet(f'{H}/v12/feat_v3/val{c}.parquet',columns=['rowid','clicked','u']); return d['clicked'].to_numpy(),d['rowid'].to_numpy(),d['u'].to_numpy()
def build(rebuild=False):
    """returns {cut: dict(y, u, raw{name:vec}, rk{name:vec})}; cached in cb_blend/oof_c{cut}.npz"""
    data={}
    for c in CUTS:
        p=f'{OUT}/oof_c{c}.npz'; y,rid,u=labels(c); n=len(y)
        names=[m for m in MEMBERS if c in MEMBERS[m]]
        cache=dict(np.load(p,allow_pickle=True)) if (os.path.exists(p) and not rebuild) else {}
        for m in names:
            if f'raw_{m}' not in cache:
                v=np.asarray(MEMBERS[m][c](),np.float64); assert v.shape==(n,) and np.isfinite(v).all(),(m,c,v.shape,n); cache[f'raw_{m}']=v.astype(np.float32)
        cache['y']=y; cache['rowid']=rid; np.savez(p,**{k:v for k,v in cache.items()})
        data[c]=dict(y=y,u=u,raw={m:cache[f'raw_{m}'].astype(np.float64) for m in names})
        data[c]['rk']={m:rk(v) for m,v in data[c]['raw'].items()}
    return data
def check_alignment(data):
    rows=[]
    for c in CUTS:
        d=data[c]
        for m,v in d['raw'].items():
            a=AUC(d['y'],v); rep=REPORTED.get(m,{}).get(c)
            rows.append((m,c,len(v),a,rep,None if rep is None else a-rep, 'OK' if rep is not None and abs(a-rep)<=TOL.get(m,1.6e-4) else ('NO-REF' if rep is None else 'MISMATCH')))
    return rows
# ---------------- weight search ----------------
def sc(w,X): return X@w
def fit_w(Xs,ys,names,obj='mean',start=None,reg_to=None,lam=0.0,seed=0):
    """nonneg weights summing to 1 maximising mean over folds of AUC(sum w_i rank_i) - lam*||w-reg_to||^2 (softmax param, Nelder-Mead from several starts)."""
    k=len(names); rs=np.random.RandomState(seed)
    def f(th):
        w=np.exp(th-th.max()); w/=w.sum(); a=np.mean([AUC(y,X@w) for X,y in zip(Xs,ys)])
        return -(a-(lam*np.sum((w-reg_to)**2) if reg_to is not None else 0))
    starts=[np.zeros(k)]+([np.log(np.maximum(start,1e-3))] if start is not None else [])+[rs.randn(k)*.5 for _ in range(1)]
    best=None
    for s in starts:
        r=minimize(f,s,method='Nelder-Mead',options=dict(xatol=2e-3,fatol=2e-7,maxiter=120*k,adaptive=True))
        if best is None or r.fun<best.fun: best=r
    w=np.exp(best.x-best.x.max()); w/=w.sum(); w[w<5e-3]=0; return w/w.sum()
def lr_stack(Xs,ys,names,C=1e-3):
    """logistic regression on ranks (strong L2). Pooled over given folds, coefs are used as scores; returned normalised to sum 1 (negatives allowed)."""
    from sklearn.linear_model import LogisticRegression
    X=np.vstack(Xs); y=np.concatenate(ys); Xc=X-X.mean(0)
    m=LogisticRegression(C=C,max_iter=200).fit(Xc*4,y); w=m.coef_[0]; return w/np.abs(w).sum()*np.sign(w.sum())
def XY(data,cut,names): return np.column_stack([data[cut]['rk'][m] for m in names]),data[cut]['y']
def dget(d,names): return np.column_stack([d['rk'][m] for m in names])
def rho_vs(data,cut,w,names,ref='M',samp=3):
    s=dget(data[cut],names)@w; return spearmanr(s[::samp],data[cut]['rk'][ref][::samp])[0]
def user_group_delta(data,cut,s_new,s_base,G=8):
    """AUC delta (new-base) per user-hash subset of a fold -> mean,sd over G subsets (uncertainty indicator, not a CI)"""
    u=data[cut]['u']; g=(pl.Series(u).hash(seed=3).to_numpy()%G); y=data[cut]['y']; d=[AUC(y[g==i],s_new[g==i])-AUC(y[g==i],s_base[g==i]) for i in range(G)]
    return float(np.mean(d)),float(np.std(d,ddof=1))
def run(members_sel,data,log):
    base_w={'z':.4,'M':.6}
    res=[]    # rows: candidate, weights dict, {cut:auc}, heldout flag text
    def add(cand,names,w,aucs,held,extra=''): res.append(dict(cand=cand,w={n:round(float(x),3) for n,x in zip(names,w) if abs(x)>1e-9},auc=aucs,held=held,extra=extra))
    def aucs_of(names,w,cuts): return {c:AUC(data[c]['y'],dget(data[c],names)@w) for c in cuts}
    FOLDS=[23,25]
    # baselines
    for nm,names,w in [('base .4z+.6M',['z','M'],[.4,.6]),('.4z+.6A_all',['z','A'],[.4,.6]),('.3z+.3M+.4D',['z','M','D'],[.3,.3,.4]),('.3z+.3A+.4D',['z','A','D'],[.3,.3,.4])]:
        cuts=[c for c in (23,25) if all(m in data[c]['rk'] for m in names)]; add(nm,names,w,aucs_of(names,np.array(w),cuts),'n/a (fixed)')
    # raw-space reproduction of Grok's numbers
    rawrep={}
    for c in (23,25):
        z=data[c]['raw']['z']; M=data[c]['raw']['M']; A=data[c]['raw']['A']; rawrep[f'.4z+.6M raw C{c}']=AUC(data[c]['y'],.4*z+.6*M); rawrep[f'.4z+.6A raw C{c}']=AUC(data[c]['y'],.4*z+.6*A)
    z,M,D=data[25]['raw']['z'],data[25]['raw']['M'],data[25]['raw']['D']; rawrep['.3z+.3M+.4D raw C25']=AUC(data[25]['y'],.3*z+.3*M+.4*D)
    log['rawrep']=rawrep
    return res,FOLDS
def search(data,sets,log):
    out=[]
    for tag,names in sets:
        k=len(names); folds=[c for c in (23,25) if all(m in data[c]['rk'] for m in names)]
        if len(folds)<2: continue
        Xs={c:XY(data,c,names)[0] for c in folds}; ys={c:data[c]['y'] for c in folds}
        b0=np.array([{'z':.4,'M':.6}.get(n,0.) for n in names]); eq=np.ones(k)/k
        W={}
        for fa,fb in ((23,25),(25,23)):
            W[f'fit C{fa}']=(fit_w([Xs[fa]],[ys[fa]],names,start=b0),[fb])
        W['mean-of-folds (in-sample)']=(fit_w([Xs[c] for c in folds],[ys[c] for c in folds],names,start=b0),[])
        # shrunk variants of cross-fold fits
        cand=[]
        for lab,(w,ev) in W.items():
            cand.append((lab,w,ev))
            if ev:
                cand.append((lab+' shrunk 50% to base',.5*w+.5*b0/b0.sum(),ev)); cand.append((lab+' shrunk 50% to equal',.5*w+.5*eq,ev))
        # LR stack: fit on one fold, eval on the other
        for fa,fb in ((23,25),(25,23)):
            w=lr_stack([Xs[fa]],[ys[fa]],names); cand.append((f'LR-L2 fit C{fa}',w,[fb]))
        for lab,w,ev in cand:
            held={c:AUC(ys[c],Xs[c]@w) for c in ev}; out.append(dict(set=tag,names=names,label=lab,w=w,held=held,
                 base={c:AUC(ys[c],Xs[c][:, [names.index('z'),names.index('M')]]@np.array([.4,.6])) for c in ev}))
            print(tag,lab,np.round(w,3),{c:round(v,5) for c,v in held.items()},flush=True)
    return out
if __name__=='__main__':
    t0=time.time(); rebuild='--rebuild' in sys.argv
    data=build(rebuild); log={}
    rows=check_alignment(data); log['align']=rows
    for r in rows: print('ALIGN',r)
    res,_=run(None,data,log)
    for r in res: print('BASE',r['cand'],r['auc'])
    print('RAW',log['rawrep'])
    sets=[('z,M',['z','M']),('z,M,big',['z','M','big']),('z,M,big,D',['z','M','big','D']),('z,M,big,ref',['z','M','big','ref']),('z,M,big,D,ref',['z','M','big','D','ref']),('z,M,D',['z','M','D']),('z,M,D,ref',['z','M','D','ref']),
          ('z,M,cat',['z','M','cat']),('z,M,graph',['z','M','graph']),('z,M,big,D,cat',['z','M','big','D','cat']),('z,M,big,D,graph',['z','M','big','D','graph']),
          ('z,M,big,D,cat,graph',['z','M','big','D','cat','graph']),('z,M,D,graph',['z','M','D','graph'])]
    if '--members' in sys.argv: sets=[('custom',sys.argv[sys.argv.index('--members')+1].split(','))]
    S=search(data,sets,log)
    import pickle; pickle.dump(dict(log=log,res=res,S=S),open(f'{OUT}/search_out.pkl','wb')); print('done',round(time.time()-t0),'s')
