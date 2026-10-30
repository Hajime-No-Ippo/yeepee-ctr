"""Step 5: regularization grid on the grid tier (smallest tier >= max(N0,100)); each config: early stop on C23 -> rounds; fixed rounds on C25 seed 0. Top 2 (+dwbig) confirmed on C23/C25/C26 seed 0.
Deviation (compute): full 16-config grid on ONE tier; other tiers reuse the winning config (rounds re-picked per tier by C23 ES in s6)."""
import os; os.environ.update(OMP_NUM_THREADS='4',POLARS_MAX_THREADS='4')
from lib20 import *
import itertools, numpy as np
T=json.load(open(f'{H}/out/tiers_chosen.json')); ranked=json.load(open(f'{H}/out/ranked_feats.json'))
gt=min(t for t in T['tiers'] if t>=max(T['N0_flat'],100)) if any(t>=max(T['N0_flat'],100) for t in T['tiers']) else max(T['tiers']); feats=ranked[:gt]
R=f'{H}/results.jsonl'; rows=[json.loads(l) for l in open(R)] if os.path.exists(R) else []; done={r['tag'] for r in rows if r.get('stage')=='grid'}
grid=[dict(num_leaves=a,min_data_in_leaf=b,lambda_l1=c,lambda_l2=d,feature_fraction=0.6,bagging_fraction=0.8,bagging_freq=1,learning_rate=0.03) for a,b,c,d in itertools.product((63,127),(200,500),(0,1),(1,10))]
res_c25={}
for i,P in enumerate(grid):
    tag=f'g{i}'
    if tag in done: continue
    wait_idle(); r23,m23,_=fit(23,feats,P,2500,0,4,es=True,tag=tag+'_es'); nr=max(200,r23['best_iter']); del m23
    wait_idle(); r25,m25,_=fit(25,feats,P,nr,0,4,tag=tag); r25.update(stage='grid',tag=tag,grid_tier=gt,rounds_es_c23=nr,auc_c23_es=r23['auc'],P=P); log(R,r25); print(json.dumps(r25),flush=True)
rows=[r for r in map(json.loads,open(R)) if r.get('stage')=='grid' and r['tag']!='gdw']
top=sorted(rows,key=lambda r:-r['auc'][str(r['rounds'])])[:2]
conf=[]
cands=[('dwbig',dict(),1000)]+[(r['tag'],r['P'],r['rounds_es_c23']) for r in top]
for tag,P,nr in cands:
    rr={}
    for cut in (23,25,26):
        k=f'conf_{tag}_{cut}'
        ex=[json.loads(l) for l in open(R)]
        f=[e for e in ex if e.get('stage')=='gridconf' and e['tag']==k]
        if f: rr[cut]=f[0]; continue
        wait_idle(); r,_,_=fit(cut,feats,P,nr,0,4,tag=k); r.update(stage='gridconf',tag=k,grid_tier=gt,P=P); log(R,r); rr[cut]=r; print(json.dumps(r),flush=True)
    conf.append(dict(tag=tag,P=P,rounds=nr,auc={c:rr[c]['auc'][str(nr)] for c in rr},mean=float(np.mean([rr[c]['auc'][str(nr)] for c in rr]))))
best=max([c for c in conf if c['tag']!='dwbig'],key=lambda c:c['mean']); dw=[c for c in conf if c['tag']=='dwbig'][0]
json.dump(dict(grid_tier=gt,best=best,dwbig_on_tier=dw,best_not_worse_than_dwbig=best['mean']>=dw['mean']-1e-4,confirmed=conf),open(f'{H}/out/best_params.json','w'),indent=1)
print('best',best['tag'],best['mean'],'dwbig',dw['mean'])
