"""Step 6: per tier: best params (out/best_params.json), rounds by early stopping on C23 (seed 0), then FIXED rounds on C23/C25/C26 x seeds 0-2. -> out/tiers.csv (484 reference first row), feats_<tier>.json"""
import os; os.environ.update(OMP_NUM_THREADS='4',POLARS_MAX_THREADS='4')
from lib20 import *
import numpy as np, pandas as pd
ranked=json.load(open(f'{H}/out/ranked_feats.json')); T=json.load(open(f'{H}/out/tiers_chosen.json')); B=json.load(open(f'{H}/out/best_params.json'))['best']; PRM=B['P']
ref=json.load(open(f'{H}/out/ref.json')); R=f'{H}/results.jsonl'
def rows(): return [json.loads(l) for l in open(R)]
def tier_row(name,n,params,res):
    d=dict(tier=name,n_feat=n,params=json.dumps(params))
    for c in (23,25,26): d[f'auc_c{c}']=float(np.mean([r['auc'][str(r['rounds'])] for r in res if r['cut']==c]))
    m=[d[f'auc_c{c}'] for c in (23,25,26)]; d['mean']=float(np.mean(m)); d['std']=float(np.std(m))
    d['seed_std']=float(np.mean([np.std([r['auc'][str(r['rounds'])] for r in res if r['cut']==c]) for c in (23,25,26)]))
    for c in (23,25,26): d[f'fit_s_c{c}']=float(np.mean([r['fit_s'] for r in res if r['cut']==c]))
    return d
out=[dict(tier='ref484',n_feat=484,params='dwbig 1000r',auc_c23=ref['c23']['mean'],auc_c25=ref['c25']['mean'],auc_c26=ref['c26']['mean'],mean=ref['mean_over_folds'],std=ref['fold_std'],
  seed_std=float(np.mean([ref[f'c{c}']['seed_std'] for c in (23,25,26)])),fit_s_c23=np.mean(ref['c23']['fit_s']),fit_s_c25=np.mean(ref['c25']['fit_s']),fit_s_c26=np.mean(ref['c26']['fit_s']))]
for t in T['tiers']:
    feats=ranked[:t]; json.dump(feats,open(f'{H}/out/feats_tier{t}.json','w'))
    ex=[r for r in rows() if r.get('stage')=='tier' and r['tier']==t]
    es=[r for r in rows() if r.get('stage')=='tier_es' and r['tier']==t]
    if es: nr=es[0]['rounds_es']
    else:
        wait_idle(); r,_,_=fit(23,feats,PRM,2500,0,4,es=True,tag=f'tier{t}_es'); nr=max(200,r['best_iter']); r.update(stage='tier_es',tier=t,rounds_es=nr); log(R,r); print(json.dumps(r),flush=True)
    for cut in (25,23,26):
        for seed in (0,1,2):
            if any(e['cut']==cut and e['seed']==seed for e in ex): continue
            wait_idle(); r,mdl,(Xv,yv,fl)=fit(cut,feats,PRM,nr,seed,4,tag=f'tier{t}'); r.update(stage='tier',tier=t)
            np.save(f'{H}/ens/tier{t}_c{cut}_s{seed}.npy',mdl.predict(Xv,raw_score=True).astype(np.float32)); log(R,r); ex.append(r); print(json.dumps(r),flush=True)
    out.append(tier_row(f'tier{t}',t,dict(PRM,rounds=nr),ex))
    pd.DataFrame(out).to_csv(f'{H}/out/tiers.csv',index=False)
out.append(dict(tier='AV-dropped variant',n_feat=None,params='NOT a tier: flagged features are never dropped from tiers; A/B of top-5 impression drop is run in s7_avdrop (out/avdrop.json), not adopted'))
pd.DataFrame(out).to_csv(f'{H}/out/tiers.csv',index=False); print(pd.DataFrame(out).to_string())
