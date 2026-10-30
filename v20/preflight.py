"""Fail-loud prerequisite check (no heavy compute): imports, data paths for every table/extras the fits need, static inputs, stage scripts present."""
import sys,os,glob,json; sys.path.insert(0,'/workspace/huawei/v20')
H='/workspace/huawei/v20'; bad=[]
import polars, lightgbm, sklearn, scipy, numpy, pandas
import cl, lib20
tabs=['f1','f2','f3','f4','f5','f6','val23','val25','val26','eval']
for n in tabs:
    if not os.path.exists(f'{cl.P3}/{n}.parquet'): bad.append(f'{cl.P3}/{n}.parquet')
    if not os.path.exists(f'{cl.P2}/{cl.V2_OF[n]}.parquet'): bad.append(f'{cl.P2}/{cl.V2_OF[n]}.parquet')
    if n in ('f6',): continue
    g=glob.glob(f'{cl.CACHE}/{n}_*.parquet')
    if len(g)!=1 or not os.path.exists(g[0]+'.json'): bad.append(('cache_mx',n,g))
    for sub in lib20.EXTRA:
        if sub in cl.BASE: continue
        d=[p for p in (f'/workspace/huawei/v19/{sub}',f'/workspace/huawei/v17cooc/{sub}',f'/workspace/huawei/v12x/{sub}',f'{cl.D}/{sub}') if os.path.isdir(p)]
        if not d or not os.path.exists(d[0]+f'/{n}.parquet'): bad.append((sub,n,d))
for p in ['/workspace/huawei/example_submission.csv',f'{H}/gain_rank.json',f'{H}/feats_ref484.json',f'{H}/av_flagged.json',f'{H}/out/av_flagged_0431.json',f'{H}/out/redundancy.json',f'{H}/ens/ref484_c25_s0.npy']+[f'{H}/{s}.py' for s in 's0_ref s2_importance s4_curve s5_grid s6_tiers s7_avdrop s8_submit report'.split()]+[f'{H}/decide.py']:
    if not os.path.exists(p): bad.append(p)
a=json.load(open(f'{H}/av_flagged.json')); assert a['n_flagged']==44 and a['overall_av_auc']>0.99, 'av_flagged.json not the authoritative 0431 result'
f=json.load(open(f'{H}/feats_ref484.json')); assert len(f)==484
for c in ['ev_it_imp_mean','ev_it_imp_max','slate_imp_coh','v2_ip_per_imp','v2_i_imp']: assert c in f, c
k=json.load(open(f'{H}/out/redundancy.json')); assert k['n_kept']==401 and len(k['kept'])==401, k['n_kept']
if bad: print('PREFLIGHT FAIL, missing:',bad); sys.exit(3)
print('PREFLIGHT OK')
