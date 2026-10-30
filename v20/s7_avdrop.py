"""Step 7 (Eric-approved A/B ONLY): refit the decided tier and the ref484 set with the 5 top impression features removed (those present),
same params/rounds as their undropped base, seeds 0-2, cuts C23/C25/C26 -> results.jsonl stage 'avdrop' (resumable), out/avdrop.json.
NOT adopted anywhere: s8_submit still uses the undropped decided tier."""
import os; os.environ.update(OMP_NUM_THREADS='4',POLARS_MAX_THREADS='4')
from lib20 import *
import lib20, numpy as np
AV5=['ev_it_imp_mean','ev_it_imp_max','slate_imp_coh','v2_ip_per_imp','v2_i_imp']; R=f'{H}/results.jsonl'; CUTS=(23,25,26); SEEDS=(0,1,2)
def rows(): return [json.loads(l) for l in open(R)]
D=json.load(open(f'{H}/out/decision.json')).get('chosen')
sets=[]   # (name, feats-or-None(=ref484 all), params, rounds, base-rows filter)
if D:
    t=D['n_feat']; feats=json.load(open(f'{H}/out/feats_tier{t}.json')); B=json.load(open(f'{H}/out/best_params.json'))['best']['P']
    nr=[r for r in rows() if r.get('stage')=='tier_es' and r['tier']==t][0]['rounds_es']
    sets.append((f'tier{t}',feats,B,nr,lambda r,t=t: r.get('stage')=='tier' and r.get('tier')==t))
sets.append(('ref484',None,{},1000,lambda r: r.get('stage')=='ref'))   # ref484 LAST per cut (it compacts the cached matrix in place)
def compact_drop(cut, drop):
    """memory-lean in-place column removal from the cached train matrix of this cut (avoids a 2nd 4+GB copy)."""
    Xt,yt,Xv,yv,fl=lib20.load(cut); keep=[i for i,c in enumerate(fl) if c not in drop]
    if len(keep)==len(fl): return
    n,k=len(Xt),len(keep); flat=Xt.reshape(-1); ch=100000
    for r0 in range(0,n,ch):
        r1=min(n,r0+ch); blk=Xt[r0:r1][:,keep]; flat[r0*k:r1*k]=blk.reshape(-1); del blk
    Xt2=flat[:n*k].reshape(n,k); lib20._C[cut]=(Xt2,yt,Xv[:,keep],yv,[fl[i] for i in keep]); gc.collect()
done={(r['set'],r['cut'],r['seed']) for r in rows() if r.get('stage')=='avdrop'}
removed={}
for cut in CUTS:
    for name,feats,P,nr,_ in sets:
        todo=[s for s in SEEDS if (name,cut,s) not in done]
        if not todo: continue
        if feats is None:
            wait_idle(); fl=lib20.load(cut)[4]; removed[name]=[c for c in AV5 if c in fl]; compact_drop(cut,set(AV5)); f=None
        else:
            f=[c for c in feats if c not in AV5]; removed[name]=[c for c in AV5 if c in feats]
        for s in todo:
            wait_idle(); r,mdl,_=fit(cut,f,P,nr,s,4,tag=f'avdrop_{name}'); r.update(stage='avdrop',set=name,removed=removed[name],base_rounds=nr); log(R,r); done.add((name,cut,s)); print(json.dumps(r),flush=True); del mdl; gc.collect()
# summary
allr=rows(); out=dict(note='A/B ONLY. Not adopted; s8_submit uses the undropped decided tier.',dropped_candidates=AV5,
  rule="drop helps iff mean_over_folds >= base_mean-0.0005 AND (mean_over_folds > base_mean OR fold_std < base_fold_std); else keep features",sets={})
for name,feats,P,nr,flt in sets:
    base=[r for r in allr if flt(r)]; ab=[r for r in allr if r.get('stage')=='avdrop' and r['set']==name]
    def agg(rs,rounds):
        pf={c:[r['auc'][str(r['rounds'])] for r in rs if r['cut']==c] for c in CUTS}
        m={c:float(np.mean(v)) for c,v in pf.items()}; mm=float(np.mean(list(m.values()))); sd=float(np.std(list(m.values())))
        return dict(per_fold_mean=m,per_fold_n_seeds={c:len(v) for c,v in pf.items()},mean_over_folds=mm,fold_std=sd,seed_std=float(np.mean([np.std(v) for v in pf.values()])))
    e=dict(n_feat_base=(len(feats) if feats else 484),rounds=nr,removed=sorted({c for r in ab for c in r['removed']}))
    if all(sum(1 for r in ab if r['cut']==c)==len(SEEDS) for c in CUTS) and all(sum(1 for r in base if r['cut']==c)>=len(SEEDS) for c in CUTS):
        b=agg(base,nr); a=agg(ab,nr); d=a['mean_over_folds']-b['mean_over_folds']
        helps=bool(a['mean_over_folds']>=b['mean_over_folds']-0.0005 and (d>0 or a['fold_std']<b['fold_std']))
        e.update(base=b,dropped=a,delta_mean=d,delta_fold_std=a['fold_std']-b['fold_std'],delta_per_fold={c:a['per_fold_mean'][c]-b['per_fold_mean'][c] for c in CUTS},
                 verdict='drop helps' if helps else 'keep features',n_feat_dropped=e['n_feat_base']-len(e['removed']))
    else: e['verdict']='INCOMPLETE'
    out['sets'][name]=e
json.dump(out,open(f'{H}/out/avdrop.json','w'),indent=1,default=str); print(json.dumps({k:v.get('verdict') for k,v in out['sets'].items()}))
