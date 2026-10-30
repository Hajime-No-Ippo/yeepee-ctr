import json,numpy as np
H='/workspace/huawei/v19/ens/'
runs=['nmAs_c23_s0','nmAs_c25_s0']   # 484-feature variant-A config only (B files excluded by data rule)
G=[json.load(open(H+r+'_gain.json')) for r in runs]
feats=list(G[0]); assert all(list(g)==feats for g in G) and len(feats)==484
N=np.array([[g[f]/sum(g.values()) for f in feats] for g in G])
m=N.mean(0); cv=N.std(0)/(m+1e-12)
rk=np.argsort(np.argsort(-N,1),1)+1
order=np.argsort(-m)
out=[dict(rank=i+1,feature=feats[j],gain_norm_mean=float(m[j]),gain_cv=float(cv[j]),rank_c23=int(rk[0,j]),rank_c25=int(rk[1,j])) for i,j in enumerate(order)]
json.dump(dict(source=runs,note='seed 0 only, 2 folds; refined in step 2',ranked=out),open('gain_rank.json','w'),indent=1)
print(sum(m[order[:50]]),sum(m[order[:100]]),sum(m[order[:250]]))
for o in out[:15]: print(o['rank'],o['feature'],round(o['gain_norm_mean'],4),o['rank_c23'],o['rank_c25'])
json.dump(feats,open('feats_ref484.json','w'))
