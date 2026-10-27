import numpy as np, polars as pl
from sklearn.metrics import roc_auc_score
R='/workspace/huawei/staleness_test'
EVAL_MIX=np.array([0.67,0.165,0.165])
def bucket(s): return np.digitize(s,[24,48])
def rew(stale):
    b=bucket(stale); share=np.bincount(b,minlength=3)/len(b); tgt=EVAL_MIX.copy()
    if share[2]==0: tgt=np.array([0.67,0.33,0.0])
    return (tgt/np.where(share>0,share,1))[b]
r=pl.read_csv(f'{R}/results_raw.csv')
cols=['auc','auc_rw','auc_b0','auc_b1','auc_b2']
out=[]
for (C,v),g in r.group_by(['C','variant'],maintain_order=True):
    base=r.filter((pl.col('C')==C)&(pl.col('variant')=='base')).sort('seed')
    g=g.sort('seed'); d={'C':C,'variant':v}
    for c in cols:
        if c in g.columns and g[c].null_count()<g.height:
            d[c]=f"{g[c].mean():.4f}±{g[c].std():.4f}"
            if v!='base':
                dd=(g[c].to_numpy()-base[c].to_numpy()); d['d_'+c]=f"{dd.mean():+.4f}±{dd.std(ddof=1):.4f}"
    out.append(d)
print(pl.DataFrame(out).to_pandas().to_string())
# bootstrap (resample events) on seed-averaged predictions: diff of reweighted AUC vs base
rng=np.random.default_rng(0)
for C in (25,24):
    z=np.load(f'{R}/preds_C{C}.npz'); y=z['y']; s=z['stale']; e=z['e']; w=rew(s)
    ue,inv=np.unique(e,return_inverse=True); nE=len(ue)
    pb=z['base'].mean(0)
    for v in [k for k in z.files if k not in('y','stale','e','base')]:
        pv=z[v].mean(0); D=[]
        full=roc_auc_score(y,pv,sample_weight=w)-roc_auc_score(y,pb,sample_weight=w)
        for _ in range(100):
            cnt=np.bincount(rng.integers(0,nE,nE),minlength=nE)[inv]; ww=w*cnt
            D.append(roc_auc_score(y,pv,sample_weight=ww)-roc_auc_score(y,pb,sample_weight=ww))
        print(f'C={C} {v:8s} seed-avg d_auc_rw={full:+.4f}  bootstrap SE={np.std(D):.4f}  95%CI=[{np.percentile(D,2.5):+.4f},{np.percentile(D,97.5):+.4f}]')
