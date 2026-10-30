"""Step 8: cut-27 fit of the chosen tier (seeds 0-2, fixed rounds), logit mean -> sigmoid; validate vs example_submission.csv; NO upload."""
import os,sys; os.environ.update(OMP_NUM_THREADS='4',POLARS_MAX_THREADS='4')
from lib20 import *
import numpy as np, pandas as pd, hashlib, zipfile
from scipy.stats import spearmanr
D=json.load(open(f'{H}/out/decision.json'))['chosen']; t=D['n_feat']; tag=D['tier']; feats=json.load(open(f'{H}/out/feats_tier{t}.json'))
nr=[json.loads(l) for l in open(f'{H}/results.jsonl')]; nr=[r for r in nr if r.get('stage')=='tier_es' and r['tier']==t][0]['rounds_es']; PRM=json.load(open(f'{H}/out/best_params.json'))['best']['P']
Z=[]; fs=[]
for s in (0,1,2):
    wait_idle(); r,mdl,(Xv,yv,fl)=fit(27,feats,PRM,nr,s,4,tag=f'{tag}_c27'); Z.append(mdl.predict(Xv,raw_score=True)); fs.append(r['fit_s']); r.update(stage='sub'); log(f'{H}/results.jsonl',r); print(r,flush=True)
rid=load(27,feats) and None; from lib20 import _C; rowid=_C['rowid']; z=np.mean(Z,0); p=1/(1+np.exp(-z))
ex=pd.read_csv('/workspace/huawei/example_submission.csv'); df=pd.DataFrame({'rowid':rowid,'predicted_probability':p}).set_index('rowid').loc[ex.rowid].reset_index()
assert list(df.columns)==['rowid','predicted_probability'] and len(df)==529390 and (df.rowid.values==ex.rowid.values).all() and np.isfinite(df.predicted_probability).all() and df.predicted_probability.between(0,1).all()
name=f'{H}/out/submission_v20_{tag}.csv'; df.to_csv(name,index=False)
with zipfile.ZipFile(name+'.zip','w',zipfile.ZIP_DEFLATED) as zf: zf.write(name,os.path.basename(name))
sha=hashlib.sha256(open(name+'.zip','rb').read()).hexdigest(); open(name+'.zip.sha256','w').write(sha+'\n')
ref=pd.read_csv('/workspace/huawei/v17cooc/out/submission_v17cooc_A_trainonly.csv').set_index('rowid').loc[ex.rowid]
sp=spearmanr(df.predicted_probability,ref.predicted_probability)[0]; json.dump(dict(file=name,sha256_zip=sha,rows=len(df),spearman_vs_v17cooc_A=sp,fit_s_c27=fs,validator='PASS'),open(f'{H}/out/submission_info.json','w'),indent=1); print('validator PASS',sha,sp)
