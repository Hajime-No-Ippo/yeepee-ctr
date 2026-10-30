"""Step 0: 484-feature variant-A reference (v19 nmAs recipe: dwbig params, 1000 rounds) seeds 0-2 on C23/C25/C26 -> out/ref.json; results.jsonl incremental."""
import os; os.environ.update(OMP_NUM_THREADS='4',POLARS_MAX_THREADS='4')
from lib20 import *
import numpy as np
R=f'{H}/results.jsonl'; done={(r['stage'],r['cut'],r['seed']) for r in map(json.loads,open(R))} if os.path.exists(R) else set()
for cut in (25,23,26):
    for seed in (0,1,2):
        if ('ref',cut,seed) in done: continue
        wait_idle(); res,mdl,(Xv,yv,fl)=fit(cut,None,None,1000,seed,4,save_at=[400,700,1000],tag='ref484')
        res['stage']='ref'; np.save(f'{H}/ens/ref484_c{cut}_s{seed}.npy',mdl.predict(Xv,raw_score=True).astype(np.float32))
        if seed==0: json.dump(dict(zip(fl,mdl.feature_importance('gain').tolist())),open(f'{H}/ens/ref484_c{cut}_s0_gain.json','w'))
        log(R,res); print(json.dumps(res),flush=True)
# summary
rows=[json.loads(l) for l in open(R)]; rows=[r for r in rows if r['stage']=='ref']; out={}
for cut in (23,25,26):
    a=[r['auc']['1000'] for r in rows if r['cut']==cut]
    if a: out[f'c{cut}']=dict(per_seed=a,mean=float(np.mean(a)),seed_std=float(np.std(a)),fit_s=[r['fit_s'] for r in rows if r['cut']==cut])
if len(out)==3:
    m=[out[f'c{c}']['mean'] for c in (23,25,26)]; out['mean_over_folds']=float(np.mean(m)); out['fold_std']=float(np.std(m))
r25=[r for r in rows if r['cut']==25 and r['seed']==0]
if r25: out['check_c25_s0_vs_logged_.81751']=dict(got=r25[0]['auc']['1000'],diff=round(r25[0]['auc']['1000']-0.81751,5),within_1e4=abs(r25[0]['auc']['1000']-0.81751)<=1e-4)
json.dump(out,open(f'{H}/out/ref.json','w'),indent=1); print(json.dumps(out))
