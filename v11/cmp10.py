"""cmp10.py TAG [TAG...] [--w 0.2] [--sq DIR] [--ref trim] [--refw 0.2]: z=(b400+s500)/2, zf=(1-w)z+w*sq_logit (sq from DIR/val{c}.parquet,
default v8 feat_sqv/all500); AUC overall, by staleness bucket, staleness-reweighted (66.6/16.6/16.8). ref = v8 trim at w 0.2 with all500 sq."""
import sys, json, numpy as np, polars as pl
from sklearn.metrics import roc_auc_score as A
V8='/workspace/huawei/v8'; EV=np.array([0.6662,0.1655,0.1682]); av=sys.argv[1:]
def opt(k,d):
    if k in av: i=av.index(k); v=av[i+1]; del av[i:i+2]; return v
    return d
w=float(opt('--w','0.2')); sqd=opt('--sq',f'{V8}/feat_sqv/all500'); ref=opt('--ref','trim'); refw=float(opt('--refw','0.2')); tags=av
Y={};BK={};W={};SQ={};SQ8={}
for c in (25,26,23):
    v=pl.read_parquet(f'/workspace/huawei/v10/feat_v3/val{c}.parquet',columns=['rowid','clicked'])
    Y[c]=v['clicked'].to_numpy()
    for D_,T_ in ((sqd,SQ),(f'{V8}/feat_sqv/all500',SQ8)):
        T_[c]=v.join(pl.read_parquet(f'{D_}/val{c}.parquet',columns=['rowid','sq_logit']),on='rowid',how='left',maintain_order='left')['sq_logit'].to_numpy()
    st=pl.read_parquet(f'/workspace/huawei/v10/feat_tmB2/val{c}.parquet')['tmR_stale_h'].to_numpy(); b=np.digitize(st,[24,48]); BK[c]=b
    sh=np.bincount(b,minlength=3)/len(b); pres=sh>0; tgt=np.where(pres,EV,0); tgt/=tgt.sum(); W[c]=(tgt/np.where(pres,sh,1))[b]
def z(tag,c,w,sq):
    b=np.load(f'/workspace/huawei/v10/ens/{tag}_b_c{c}_s0.npy')[:,1]; s=np.load(f'/workspace/huawei/v10/ens/{tag}_s_c{c}_s0.npy')[:,2]; return (1-w)*(b+s)/2+w*sq[c]
def stats(z,c):
    y=Y[c]; b=BK[c]; r=dict(auc=A(y,z),rw=A(y,z,sample_weight=W[c]))
    for k,nm in enumerate(('b0_24','b24_48','b48p')):
        m=b==k; r[nm]=A(y[m],z[m]) if m.sum()>1000 else np.nan
    return r
if __name__=='__main__':
  for tag in tags:
    for c in (25,23,26):
      try: zz=z(tag,c,w,SQ)
      except FileNotFoundError: continue
      s=stats(zz,c); s0=stats(z(ref,c,refw,SQ8),c)
      r=dict(tag=tag,w=w,sq=sqd.split('/')[-1],cut=c,**{k:round(v,5) for k,v in s.items()},**{'d_'+k:round(s[k]-s0[k],5) for k in s})
      print(json.dumps(r),flush=True); open('cmp10_results.jsonl','a').write(json.dumps(r)+'\n')
