"""Validate candidate drop sets vs S baseline. usage: run_val.py CUT SEEDS(comma) sets.json   (sets.json: {name:[cols]} ; 'S' = baseline added automatically)
Full folds, dwbig recipe (1000 rounds), variant A, S already dropped. Logs val_log.jsonl (resumable)."""
import sys; sys.path.insert(0,'/workspace/huawei/dw_featcut/null')
from lib import *
cut=int(sys.argv[1]); seeds=[int(s) for s in sys.argv[2].split(',')]; sets={'S':[]}; sets.update(json.load(open(sys.argv[3])))
t0=time.time(); X,y,Xv,yv,fl=load(cut,S_RE); print('loaded',X.shape,round(time.time()-t0),flush=True)
LOG=f'{OUT}/val_log.jsonl'; done={(r['cut'],r['set'],r['seed']) for r in map(json.loads,open(LOG))} if os.path.exists(LOG) else set()
for sd in seeds:
    for name,drop in sets.items():
        if (cut,name,sd) in done: continue
        ds_=set(drop); ix=[i for i,c in enumerate(fl) if c not in ds_]; assert len(ix)==len(fl)-len(ds_)
        t1=time.time(); ds=lgb.Dataset(X[:,ix],y.astype(np.float32),feature_name=[fl[i] for i in ix],free_raw_data=True,params=dict(max_bin=255,force_col_wise=True)).construct()
        mdl=lgb.train({**P,'num_threads':4,'seed':42+sd},ds,1000); fit=round(time.time()-t1)
        auc=roc_auc_score(yv,mdl.predict(Xv[:,ix],raw_score=True)); del ds,mdl; gc.collect()
        log(LOG,dict(cut=cut,set=name,seed=sd,nf=len(ix),auc=round(auc,6),fit_s=fit)); print(cut,name,sd,len(ix),round(auc,5),fit,'s',flush=True)
print('DONE',round(time.time()-t0),flush=True)
