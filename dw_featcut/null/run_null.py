"""Null-importance run: real + K block-shuffled-label fits on C25 (full train f1-f4, S already dropped), dwbig recipe, variant A.
Label null = permute the 10-row event label blocks across events (keeps per-event label structure, breaks feature<->label link)."""
import sys; sys.path.insert(0,'/workspace/huawei/dw_featcut/null')
from lib import *
K=int(sys.argv[1]) if len(sys.argv)>1 else 5; cut=25; t0=time.time()
X,y,Xv,yv,fl=load(cut,S_RE); print('loaded',X.shape,Xv.shape,round(time.time()-t0),'s',flush=True)
ds=lgb.Dataset(X,y.astype(np.float32),feature_name=fl,free_raw_data=True,params=dict(max_bin=255,force_col_wise=True)).construct(); del X; gc.collect(); print('ds',round(time.time()-t0),flush=True)
done={r['run'] for r in map(json.loads,open(f'{OUT}/null_log.jsonl'))} if os.path.exists(f'{OUT}/null_log.jsonl') else set()
yb=y.reshape(-1,10)
for run in ['real']+[f'null{k}' for k in range(K)]:
    if run in done: continue
    t1=time.time(); lab=y if run=='real' else yb[np.random.RandomState(1000+int(run[4:])).permutation(len(yb))].reshape(-1)
    ds.set_label(lab.astype(np.float32)); seed=42 if run=='real' else 142+int(run[4:])
    mdl=lgb.train({**P,'num_threads':4,'seed':seed},ds,1000)
    g=dict(zip(fl,mdl.feature_importance('gain').tolist())); s=dict(zip(fl,mdl.feature_importance('split').tolist()))
    json.dump(dict(gain=g,split=s),open(f'{OUT}/imp_{run}.json','w'))
    auc=round(roc_auc_score(yv,mdl.predict(Xv,raw_score=True)),5) if run in('real','null0') else None
    log(f'{OUT}/null_log.jsonl',dict(run=run,fit_s=round(time.time()-t1),auc_val25=auc,label_mean=float(lab.mean()),nf=len(fl))); print(run,round(time.time()-t1),'s auc',auc,flush=True)
print('DONE',round(time.time()-t0),flush=True)
