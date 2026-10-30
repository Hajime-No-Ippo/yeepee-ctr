"""v20 shared trainer. Variant A only (feat_Ap = coA, feat_NAs). Time folds: cut 23/25/26 as xlgb17 (train f_k tables, score val{cut}, row-level global AUC).
Never touches v19 outputs. Optional adversarial-validation hook is in av_optional.py (NOT run)."""
import sys, os, re, json, gc, time
sys.path.insert(0,'/workspace/huawei/v20'); sys.dont_write_bytecode=True
import numpy as np, lightgbm as lgb
from cl import matrix
from sklearn.metrics import roc_auc_score
H='/workspace/huawei/v20'; DROP='^sq_(pdot|pcos|mdot|mcos)$'
EXTRA=('feat_lr2','feat_tt','feat_pp','feat_sqC','feat_dw','feat_Ap','feat_NAs')
assert not any('B' in e.replace('feat_NAs','') for e in EXTRA)   # variant-A guard
DWBIG=dict(objective='binary',learning_rate=0.015,num_leaves=127,min_data_in_leaf=200,feature_fraction=0.5,bagging_fraction=0.8,bagging_freq=1,lambda_l2=30.0,max_bin=255,force_col_wise=True,verbose=-1)
def trn_tables(cut): return [f'f{k}' for k in range(max(1,cut-24),cut-20)]
_C={}
def load(cut, feats=None):
    """returns Xtr, ytr, Xva, yva, featnames (restricted to feats, order of feats). Cached per (cut) in-process (full matrix, column-sliced on request)."""
    if cut not in _C:
        import polars as pl; from cl import P3
        L=[pl.scan_parquet(f'{P3}/{n}.parquet').select(pl.len()).collect().item() for n in trn_tables(cut)]; Xt=None; ys=[]; off=0
        for n,l in zip(trn_tables(cut),L):
            X,m,f=matrix(n,1.0,variant='notime',extra=EXTRA); keep=[i for i,c in enumerate(f) if not re.search(DROP,c)]; fl=[f[i] for i in keep]
            if Xt is None: Xt=np.empty((sum(L),len(fl)),np.float32)
            assert len(X)==l; Xt[off:off+l]=X[:,keep]; off+=l; ys.append(m['clicked'].to_numpy()); del X,m; gc.collect()
        yt=np.concatenate(ys)
        Xv,mv,fv=matrix('eval' if cut==27 else f'val{cut}',1.0,variant='notime',extra=EXTRA); Xv=Xv[:,[fv.index(c) for c in fl]]
        _C.clear(); _C[cut]=(Xt,yt,Xv,mv['clicked'].to_numpy() if 'clicked' in mv.columns else np.zeros(len(Xv),np.int8),fl); _C['rowid']=mv['rowid'].to_numpy()
    Xt,yt,Xv,yv,fl=_C[cut]
    if feats is None: return Xt,yt,Xv,yv,fl
    ix=[fl.index(c) for c in feats]; return Xt[:,ix],yt,Xv[:,ix],yv,list(feats)
def fit(cut, feats=None, params=None, rounds=1000, seed=0, threads=4, save_at=None, es=False, perm=None, tag=''):
    """train one LightGBM; returns dict(auc{round}, fit_s, model, ...). es=True: early stopping on the val table of this cut (use ONLY for C23)."""
    threads=int(os.environ.get('V20_THREADS',threads)); t0=time.time(); Xt,yt,Xv,yv,fl=load(cut,feats); P=dict(DWBIG); P.update(params or {}); P.update(num_threads=threads,seed=42+seed)
    ds=lgb.Dataset(Xt,yt,feature_name=fl,free_raw_data=False,params={k:P[k] for k in ('max_bin','force_col_wise') if k in P}); t1=time.time()
    cb=[]; ev={}
    if es:
        dv=lgb.Dataset(Xv,yv,reference=ds); cb=[lgb.early_stopping(100,verbose=False),lgb.record_evaluation(ev)]; P['metric']='auc'
        mdl=lgb.train(P,ds,rounds,valid_sets=[dv],callbacks=cb)
    else: mdl=lgb.train(P,ds,rounds)
    fit_s=time.time()-t1; res=dict(cut=cut,tag=tag,seed=seed,nf=len(fl),fit_s=round(fit_s),rounds=rounds,params=params or {})
    if es: res['best_iter']=mdl.best_iteration
    sa=[mdl.best_iteration] if es else (save_at or [rounds]); res['auc']={} if cut==27 else {str(r):round(roc_auc_score(yv,mdl.predict(Xv,num_iteration=r,raw_score=True)),5) for r in sa if r<=mdl.current_iteration()}
    import resource; res['wall_s']=round(time.time()-t0); res['threads']=threads; res['peak_rss_mb_process']=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024); del ds; gc.collect(); return res,mdl,(Xv,yv,fl)
PAT=re.compile(r'ens17|xlgb|mlp17|mlp_pair|mkmf|mkcooc|ens\.py|combo|screen|v2work|v19/|v17run|v20fast|v18|mkv18|ttfeat|mkdw|seq')
def others_busy():
    """other agents' heavy python jobs: pattern match OR >120% CPU, excluding anything whose cwd/cmd is under /workspace/huawei/v20 (ours)."""
    import subprocess; me=os.getpid(); busy=[]; tot=0.0
    for ln in subprocess.run(['ps','-eo','pid,pcpu,args','--no-headers'],capture_output=True,text=True).stdout.splitlines():
        p=ln.split(None,2)
        if len(p)<3 or 'python' not in p[2] or p[0]==str(me) or 'ps -eo' in p[2]: continue
        try: cwd=os.readlink(f'/proc/{p[0]}/cwd')
        except Exception: continue
        if cwd.startswith('/workspace/huawei/v20') and 'v20fast' not in cwd or '/workspace/huawei/v20/' in p[2]: continue
        if float(p[1])>120 or ((PAT.search(p[2]) or PAT.search(cwd)) and float(p[1])>=5): busy.append(f'{p[0]}:{p[1]}%:{p[2][:80]}')
    return busy
def wait_idle(maxprint=1):
    n=0
    while True:
        b=others_busy()
        if not b: return
        if n%10==0: print(time.strftime('%T'),'guard waiting on',b[:3],flush=True)
        n+=1; time.sleep(60)
def log(path,obj): open(path,'a').write(json.dumps(obj)+'\n')
