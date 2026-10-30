import json, pandas as pd, numpy as np
H='/workspace/huawei/v20'; d=pd.read_csv(f'{H}/out/tiers.csv'); r=d.iloc[0]; t=d.iloc[1:][d.iloc[1:].n_feat.notna()]
q=[]
for _,x in t.iterrows():
    ok_m=x['mean']>=r['mean']-0.0005; ok_s=x['std']<=r['std']+0.0002   # std tie tolerance 2x seed std (~1e-4)
    q.append(dict(tier=x.tier,n_feat=int(x.n_feat),mean=x['mean'],delta_vs_ref=x['mean']-r['mean'],std=x['std'],ref_std=r['std'],mean_ok=bool(ok_m),std_ok=bool(ok_s),qualifies=bool(ok_m and ok_s)))
qq=[a for a in q if a['qualifies']]; ch=min(qq,key=lambda a:a['n_feat']) if qq else None
json.dump(dict(rule='mean>=ref_mean-0.0005 and fold_std<=ref_std(+2e-4 tie tol); smallest n_feat wins',candidates=q,chosen=ch),open(f'{H}/out/decision.json','w'),indent=1); print(json.dumps(ch)); raise SystemExit(0 if ch else 2)
