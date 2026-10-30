"""Steps 2-3: prefiltered set (out/redundancy.json kept) -> C23, C25 x seeds 0-2, 500 rounds (lr .03 dwbig-otherwise) -> normalized gain mean/cv;
permutation importance (predict-only, global shuffle, 3 repeats) for top-300 by gain on seed-0 models; out/importance.csv, out/stable_feats.json (AV: NOT RUN)."""
import os; os.environ.update(OMP_NUM_THREADS='4',POLARS_MAX_THREADS='4')
from lib20 import *
import numpy as np, pandas as pd
keep=json.load(open(f'{H}/out/redundancy.json'))['kept']; R=f'{H}/results.jsonl'
PRM=dict(learning_rate=0.03); NR=500; G={}; PM={}
done={(r['stage'],r['cut'],r['seed']) for r in map(json.loads,open(R))} if os.path.exists(R) else set()
for cut in (23,25):
    for seed in (0,1,2):
        gp=f'{H}/ens/imp_c{cut}_s{seed}_gain.json'
        if os.path.exists(gp) and not (seed==0 and not os.path.exists(f'{H}/out/perm_c{cut}.json')): G[(cut,seed)]=json.load(open(gp)); continue
        wait_idle(); res,mdl,(Xv,yv,fl)=fit(cut,keep,PRM,NR,seed,4,tag='imp'); res['stage']='imp'
        g=dict(zip(fl,mdl.feature_importance('gain').tolist())); json.dump(g,open(gp,'w')); G[(cut,seed)]=g; log(R,res); print(json.dumps(res),flush=True)
        if seed==0 and not os.path.exists(f'{H}/out/perm_c{cut}.json'):
            tot=sum(g.values()); top=sorted(g,key=lambda c:-g[c])[:300]
            # event-aligned 250k-row val subsample (events = consecutive rows in tens); same rows for baseline and all shuffles
            rs=np.random.RandomState(5); ne=len(yv)//10; ev=np.sort(rs.choice(ne,min(ne,25000),replace=False)); rows=(ev[:,None]*10+np.arange(10)).ravel()
            Xs=Xv[rows].copy(); ys=yv[rows]; base=roc_auc_score(ys,mdl.predict(Xs,raw_score=True)); out={}; t0=time.time()
            for i,c in enumerate(top):
                wait_idle(); j=fl.index(c); orig=Xs[:,j].copy(); d=[]
                for rep in range(3):
                    Xs[:,j]=orig[np.random.RandomState(100+rep).permutation(len(orig))]; d.append(base-roc_auc_score(ys,mdl.predict(Xs,raw_score=True)))
                Xs[:,j]=orig; out[c]=dict(mean=float(np.mean(d)),std=float(np.std(d)))
                if i%25==0: print('perm',cut,i,round(time.time()-t0),flush=True); json.dump(dict(base=base,nrows=len(rows),res=out),open(f'{H}/out/perm_c{cut}.partial.json','w'))
            json.dump(dict(base=base,nrows=len(rows),res=out),open(f'{H}/out/perm_c{cut}.json','w')); PM[cut]=out
for cut in (23,25):
    if cut not in PM: PM[cut]=json.load(open(f'{H}/out/perm_c{cut}.json'))['res']
rows=[]
N={k:{c:v/sum(g.values()) for c,v in g.items()} for k,g in G.items()}
for c in keep:
    a=np.array([N[k].get(c,0) for k in sorted(N)]); r=dict(feature=c,gain_mean=a.mean(),gain_cv=a.std()/(a.mean()+1e-12))
    for cut in (23,25):
        m=np.mean([N[(cut,s)].get(c,0) for s in (0,1,2)]); r[f'_g{cut}']=m
        p=PM[cut].get(c); r[f'perm_drop_c{cut}']=p['mean'] if p else np.nan; r[f'perm_std_c{cut}']=p['std'] if p else np.nan
    rows.append(r)
df=pd.DataFrame(rows)
for cut in (23,25): df[f'rank_c{cut}']=df[f'_g{cut}'].rank(ascending=False,method='first').astype(int)
df=df.drop(columns=['_g23','_g25']).sort_values('gain_mean',ascending=False); df.to_csv(f'{H}/out/importance.csv',index=False)
# step 3 stability filter
uns=df[(df.gain_cv>1.0)|(((df.rank_c23<=200)&(df.rank_c25>400))|((df.rank_c25<=200)&(df.rank_c23>400)))]
stable=[c for c in df.feature if c not in set(uns.feature)]
json.dump(dict(n_prefiltered=len(keep),n_unstable=len(uns),unstable=uns.feature.tolist(),stable=stable),open(f'{H}/out/stable_feats.json','w'),indent=1)

print('stable',len(stable),'unstable',len(uns))
