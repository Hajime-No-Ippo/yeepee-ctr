"""Section-9 screen: python fsx.py TAG CUT [SEED]; base = 490 (484 + feat_tmx), TAG = +-joined blocks from BLK, model tag saved as fsx_{TAG}. Same recipe as fs490.py.
Same recipe as v20/s0_ref.py (lib20 DWBIG params, 1000 rounds, seed 42+SEED) but 2 threads and Dataset freed after binning. Train-only: f-tables < cutoff, scored on val{CUT}. Never reads eval."""
import os,sys,json,time,gc
os.environ['V20_THREADS']='2'
sys.path.insert(0,'/workspace/huawei/v20')
import numpy as np, lightgbm as lgb, lib20
from sklearn.metrics import roc_auc_score as AUC
tag,cut=sys.argv[1],int(sys.argv[2]); seed=int(sys.argv[3]) if len(sys.argv)>3 else 0
H='/workspace/huawei/research/blending_v1_scripts/out'; t0=time.time()
BLK=dict(gf=('feat_s9_gf',),ttbag=('feat_s9_ttbag',),tth=('feat_s9_tth',),ttr=('feat_s9_ttr',),coAx=('feat_A',),prtm=('feat_prop','feat_tm'),ttsc=('feat_s9_ttsc',))
ex=()
for b in tag.split('+'):
    if b not in BLK: raise SystemExit('tag')
    ex+=BLK[b]
lib20.EXTRA=lib20.EXTRA+('feat_tmx',)+ex
Xt,yt,Xv,yv,fl=lib20.load(cut); print(tag,cut,'loaded',Xt.shape,round(time.time()-t0),flush=True)
ref=json.load(open('/workspace/huawei/v20/feats_ref484.json')); assert set(ref)<=set(fl); xc=[c for c in fl if c not in ref]; print('extra cols:',len(xc),xc,flush=True)
assert 'tm_inh' in fl
P=dict(lib20.DWBIG); P.update(num_threads=2,seed=42+seed)
ds=lgb.Dataset(Xt,yt,feature_name=fl,free_raw_data=True,params={k:P[k] for k in ('max_bin','force_col_wise')}); ds.construct(); del Xt; lib20._C.clear(); gc.collect()
m=lgb.train(P,ds,1000); r={}
for n in (400,700,1000): r[n]=round(AUC(yv,m.predict(Xv,num_iteration=n,raw_score=True)),5)
s=m.predict(Xv,raw_score=True).astype(np.float32); np.save(f'{H}/fsx_{tag}_c{cut}_s{seed}.npy',s)
g=dict(zip(fl,m.feature_importance('gain').tolist())); json.dump(g,open(f'{H}/fsx_{tag}_c{cut}_s{seed}_gain.json','w'))
rec=dict(tag='fsx_'+tag,nx=len(xc),cut=cut,seed=seed,nf=len(fl),auc=r,wall_s=round(time.time()-t0)); print(rec,flush=True); open(f'{H}/fs_results.jsonl','a').write(json.dumps(rec)+'\n')
