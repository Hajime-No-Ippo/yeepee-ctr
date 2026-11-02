"""fs6a.py CUT SEED [tag=f6]: ref484 + 6 features (dw_feat6/feat6) with the EXACT fs490.py recipe (lib20 DWBIG, 1000 rounds, seed 42+SEED, 2 threads). Train-only (f-tables<cutoff, scored on val{CUT}); never reads eval."""
import os,sys,json,time,gc
os.environ['V20_THREADS']='2'
sys.path.insert(0,'/workspace/huawei/v20')
import numpy as np, lightgbm as lgb, lib20
from sklearn.metrics import roc_auc_score as AUC
cut=int(sys.argv[1]); seed=int(sys.argv[2]); tag=sys.argv[3]
H='/workspace/huawei/dw_feat6/out'; os.makedirs(H,exist_ok=True); t0=time.time()
lib20.EXTRA=lib20.EXTRA+('../dw_feat6/'+sys.argv[3],)
Xt,yt,Xv,yv,fl=lib20.load(cut); ref=json.load(open('/workspace/huawei/v20/feats_ref484.json')); assert set(ref)<=set(fl)
print('loaded',Xt.shape,'extra:',[c for c in fl if c not in ref],round(time.time()-t0),flush=True); assert len(fl)==484+len([c for c in fl if c not in ref])
P=dict(lib20.DWBIG); P.update(num_threads=2,seed=42+seed)
ds=lgb.Dataset(Xt,yt,feature_name=fl,free_raw_data=True,params={k:P[k] for k in ('max_bin','force_col_wise')}); ds.construct(); del Xt; lib20._C.clear(); gc.collect()
m=lgb.train(P,ds,1000); r={}
for n in (400,700,1000): r[n]=round(AUC(yv,m.predict(Xv,num_iteration=n,raw_score=True)),5)
s=m.predict(Xv,raw_score=True).astype(np.float32); np.save(f'{H}/{tag}_c{cut}_s{seed}.npy',s)
json.dump(dict(zip(fl,m.feature_importance('gain').tolist())),open(f'{H}/{tag}_c{cut}_s{seed}_gain.json','w'))
rec=dict(tag=tag,cut=cut,seed=seed,nf=len(fl),auc=r,wall_s=round(time.time()-t0)); print(rec,flush=True); open(f'{H}/results.jsonl','a').write(json.dumps(rec)+'\n')
