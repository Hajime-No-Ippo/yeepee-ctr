"""cmp9.py TAG [TAG...] [--ref trim]: z=(b400+s500)/2, z8=0.8z+0.2sq; AUC overall, by staleness bucket (0-24h/24-48h/48h+ after cutoff),
and staleness-reweighted to eval's mix (66.6/16.6/16.8; renormalised over buckets present in the cut). deltas vs ref."""
import sys, json, numpy as np, polars as pl
from sklearn.metrics import roc_auc_score as A
V8='/workspace/huawei/v8'; EV=np.array([0.6662,0.1655,0.1682])
args=[x for x in sys.argv[1:] if not x.startswith('--')]; ref='trim'
if '--ref' in sys.argv: ref=sys.argv[sys.argv.index('--ref')+1]; args=[x for x in args if x!=ref]
def z8(tag,c):
    b=np.load(f'ens/{tag}_b_c{c}_s0.npy')[:,1]; s=np.load(f'ens/{tag}_s_c{c}_s0.npy')[:,2]; return .8*(b+s)/2+.2*SQ[c]
SQ={};Y={};BK={};W={}
for c in (25,26,23):
    v=pl.read_parquet(f'feat_v3/val{c}.parquet',columns=['rowid','clicked']).join(pl.read_parquet(f'{V8}/feat_sqv/all500/val{c}.parquet',columns=['rowid','sq_logit']),on='rowid',how='left',maintain_order='left')
    Y[c]=v['clicked'].to_numpy(); SQ[c]=v['sq_logit'].to_numpy()
    st=pl.read_parquet(f'feat_tmB2/val{c}.parquet')['tmR_stale_h'].to_numpy(); b=np.digitize(st,[24,48]); BK[c]=b
    sh=np.bincount(b,minlength=3)/len(b); pres=sh>0; tgt=np.where(pres,EV,0); tgt/=tgt.sum(); W[c]=(tgt/np.where(pres,sh,1))[b]
def stats(z,c):
    y=Y[c]; b=BK[c]; r=dict(auc=A(y,z),rw=A(y,z,sample_weight=W[c]))
    for k,nm in enumerate(('b0_24','b24_48','b48p')):
        m=b==k; r[nm]=A(y[m],z[m]) if m.sum()>1000 else np.nan
    r['b24p']=A(y[b>=1],z[b>=1]) if (b>=1).sum()>1000 else np.nan
    return r
for tag in args:
  for c in (25,26,23):
    try: z=z8(tag,c)
    except FileNotFoundError: continue
    s=stats(z,c); s0=stats(z8(ref,c),c)
    r=dict(tag=tag,ref=ref,cut=c,**{k:round(v,5) for k,v in s.items()},**{'d_'+k:round(s[k]-s0[k],5) for k in s})
    print(json.dumps(r),flush=True); open('cmp9_results.jsonl','a').write(json.dumps(r)+'\n')
