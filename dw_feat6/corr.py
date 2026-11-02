import sys,json,os,numpy as np,polars as pl
os.environ['V20_THREADS']='2'; sys.path.insert(0,'/workspace/huawei/v20')
from cl import matrix
import lib20
from scipy.stats import rankdata
X,m,fl=matrix('val25',1.0,variant='notime',extra=lib20.EXTRA)
ref=json.load(open('/workspace/huawei/v20/feats_ref484.json')); ix=[fl.index(c) for c in ref if c in fl]; names=[fl[i] for i in ix]
rng=np.random.RandomState(0); s=np.sort(rng.choice(len(X),40000,replace=False)); A=X[s][:,ix]
def rk(M):
    M=np.where(np.isfinite(M),M,np.nanmedian(M,0)); return np.apply_along_axis(rankdata,0,M)
Ar=(rk(A)-rk(A).mean(0))/ (rk(A).std(0)+1e-9)
rid=m['rowid'].to_numpy()[s]; out={}
for d in ('v12x/feat_s9_gf','v12x/feat_prop'):
    t=pl.DataFrame({'rowid':m['rowid'].to_numpy()}).join(pl.read_parquet(f'/workspace/huawei/{d}/val25.parquet'),on='rowid',how='left',maintain_order='left')
    for c in t.columns[1:]:
        a=t[c].to_numpy().astype(np.float64)[s]; a=np.where(np.isfinite(a),a,np.nanmedian(a)); r=rankdata(a); r=(r-r.mean())/(r.std()+1e-9)
        cc=(Ar*r[:,None]).mean(0); j=int(np.nanargmax(np.abs(cc))); out[c]=(float(abs(cc[j])),names[j])
for k,v in sorted(out.items(),key=lambda kv:kv[1][0]): print(f'{k:16s} maxcorr {v[0]:.3f} vs {v[1]}')
json.dump(out,open('corr484.json','w'))
