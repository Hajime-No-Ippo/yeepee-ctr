"""Step 4: rank stable features by mean permutation drop (C23,C25), ties/no-perm by gain; C25 seed 0, 500 rounds (lr .03) at N in curve -> out/curve.json, out/ranked_feats.json, out/tiers_chosen.json"""
import os; os.environ.update(OMP_NUM_THREADS='4',POLARS_MAX_THREADS='4')
from lib20 import *
import numpy as np, pandas as pd
df=pd.read_csv(f'{H}/out/importance.csv'); st=json.load(open(f'{H}/out/stable_feats.json'))['stable']; df=df[df.feature.isin(st)].copy()
df['perm']=df[['perm_drop_c23','perm_drop_c25']].mean(axis=1); df['hp']=df.perm.notna().astype(int); df['perm']=df.perm.fillna(-1)
df=df.sort_values(['hp','perm','gain_mean'],ascending=False); ranked=df.feature.tolist(); json.dump(ranked,open(f'{H}/out/ranked_feats.json','w'))
NS=[25,50,75,100,150,200,250,350,len(ranked)]; NS=sorted({min(n,len(ranked)) for n in NS}); R=f'{H}/results.jsonl'; PRM=dict(learning_rate=0.03)
done={r['n']:r for r in map(json.loads,open(R)) if r.get('stage')=='curve'} if os.path.exists(R) else {}
for n in NS:
    if n in done: continue
    wait_idle(); res,mdl,_=fit(25,ranked[:n],PRM,500,0,4,save_at=[300,500],tag=f'curve{n}'); res['stage']='curve'; res['n']=n; log(R,res); done[n]=res; print(json.dumps(res),flush=True)
    json.dump(ranked[:n],open(f'{H}/out/feats_top{n}.json','w'))
cur={n:done[n]['auc']['500'] for n in NS}; allv=cur[NS[-1]]
N0=min(n for n in NS if cur[n]>=allv-0.0003)
tiers=sorted({t for t in (50,100,150,250) if t<=len(ranked)}|({N0} if N0<250 else set()))
json.dump(dict(curve=cur,fit_s={n:done[n]['fit_s'] for n in NS},all_kept=len(ranked),N0_flat=N0,tiers=tiers),open(f'{H}/out/tiers_chosen.json','w'),indent=1)
open(f'{H}/out/curve.md','w').write('| N | C25 AUC (500r, lr.03, seed0) | fit_s |\n|---|---|---|\n'+''.join(f'| {n} | {cur[n]} | {done[n]["fit_s"]} |\n' for n in NS))
print(cur,N0,tiers)
