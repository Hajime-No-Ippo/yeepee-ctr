#!/usr/bin/env python
"""score3.py: AUC of variants on C23 (M3/val23), C25 (M5/val25), C26 (M6/f6) + mean3, plain K4 (f3..f6) mean; staleness split (<24h vs >=24h) on C23/C25; segments (cold users / cold items / never-shown pairs).
Paired Poisson bootstrap by event for delta vs ref.  usage: python score3.py --names a,b,c --ref a [--boot 200] [--tag T] [--pred-dir D]"""
import sys, json, argparse, numpy as np, polars as pl
sys.path.insert(0,'/workspace/huawei/v21trend')
from rollval_t import meta, wauc, R
ap=argparse.ArgumentParser(); ap.add_argument('--names',required=True); ap.add_argument('--ref'); ap.add_argument('--boot',type=int,default=200); ap.add_argument('--tag',default='s3'); ap.add_argument('--pred-dir'); ap.add_argument('--seg',type=int,default=1)
a=ap.parse_args(); names=a.names.split(','); ref=a.ref or names[0]; pdir=a.pred_dir or f'{R}/preds'
SETS={'C23':('val23',3),'C25':('val25',5),'C26':('f6',6),'f3':('f3',3),'f4':('f4',4),'f5':('f5',5)}
rng=np.random.default_rng(0); out={'res':{},'delta':{},'sd':{}}
def ld(n,tb,d):
    import os; p=f'{pdir}/{n}_M{d}_{tb}.npy'; return np.load(p).astype(np.float64)
def boot_auc(s,y,ev,cnt):   # cnt: B x nev
    o=np.argsort(s,kind='stable'); y=y[o]; e=ev[o]; res=np.empty(len(cnt))
    for b in range(len(cnt)):
        w=cnt[b][e]; wp=w*y; wn=w*(1-y); c=np.cumsum(wn); res[b]=(wp*(c-0.5*wn)).sum()/(wp.sum()*wn.sum())
    return res
def auc(s,y): return wauc(s,y,np.ones(len(y)))
def getseg(tb):
    f=pl.read_parquet(f'/workspace/huawei/v21trend/feat_cs/{tb}.parquet'); m=meta(tb).join(f,on='rowid',how='left',maintain_order='left'); return m
SEG={'all':None,'fresh<24h':lambda m:m['tm_stale'].to_numpy()<86400,'stale>=24h':lambda m:m['tm_stale'].to_numpy()>=86400,
     'cold_user':lambda m:m['cs_user_cold'].to_numpy()==1,'cold_item(imp24==0 or imp6==0)':lambda m:m['cs_it_noimp6'].to_numpy()==1,'cold_item(imp<20)':lambda m:m['cs_it_lowimp20'].to_numpy()==1,
     'pair_never_shown':lambda m:m['cs_ui_never_shown'].to_numpy()==1,'pair_shown':lambda m:m['cs_ui_never_shown'].to_numpy()==0,'pair_never_clk':lambda m:m['cs_ui_never_clk'].to_numpy()==1,'pair_clk':lambda m:m['cs_ui_never_clk'].to_numpy()==0}
for k,(tb,d) in SETS.items():
    m=meta(tb) if not a.seg or k not in ('C23','C25','C26') else getseg(tb)
    y=m['clicked'].to_numpy().astype(float); ev=np.unique(m['eid'].to_numpy(),return_inverse=True)[1]; nev=ev.max()+1; cnt=rng.poisson(1.0,size=(a.boot,nev)).astype(float)
    S={n:ld(n,tb,d) for n in names}
    segs=SEG if (a.seg and k in ('C23','C25','C26')) else {'all':None}
    for sg,fn in segs.items():
        msk=np.ones(len(y),bool) if fn is None else fn(m)
        if msk.sum()<2000 or y[msk].sum()<30: continue
        # event-level mask for bootstrap multiplicities: keep full event arrays, subset rows
        yy=y[msk]; ee=ev[msk]
        for n in names:
            s=S[n][msk]; out['res'].setdefault(f'{k}|{sg}',{})[n]=dict(auc=auc(s,yy),n=int(msk.sum()),share=float(msk.mean()),ctr=float(yy.mean()))
            if n==ref: continue
        dr={n:boot_auc(S[n][msk],yy,ee,cnt) for n in names}
        for n in names:
            if n==ref: continue
            d_=out['res'][f'{k}|{sg}'][n]['auc']-out['res'][f'{k}|{sg}'][ref]['auc']; out['delta'].setdefault(f'{k}|{sg}',{})[n]=d_; out['sd'].setdefault(f'{k}|{sg}',{})[n]=float((dr[n]-dr[ref]).std())
        if sg=='all': out.setdefault('_draws',{})[k]=dr
# mean3 and K4 with bootstrap
def agg(keys,n): return float(np.mean([out['res'][f'{k}|all'][n]['auc'] for k in keys]))
for lab,keys in (('mean3(C23,C25,C26)',['C23','C25','C26']),('K4plain(f3..f5,C26)',['f3','f4','f5','C26'])):
    for n in names:
        out['res'].setdefault(lab,{})[n]=agg(keys,n)
        if n!=ref:
            dd=np.mean([out['_draws'][k][n]-out['_draws'][k][ref] for k in keys],0); out['delta'].setdefault(lab,{})[n]=out['res'][lab][n]-out['res'][lab][ref]; out['sd'].setdefault(lab,{})[n]=float(dd.std())
out.pop('_draws'); json.dump(out,open(f'{R}/score3_{a.tag}.json','w'),indent=1)
print('%-34s'%'set|segment',' '.join('%24s'%n for n in names))
for key in out['res']:
    r=out['res'][key]
    if key.startswith(('mean3','K4')): row=[f'{r[n]:.5f}'+('' if n==ref else f' ({out["delta"][key][n]:+.5f}±{out["sd"][key][n]:.5f})') for n in names]; print('%-34s'%key,' '.join('%24s'%x for x in row))
    else:
        row=[f'{r[n]["auc"]:.5f}'+('' if n==ref else f' ({out["delta"][key][n]:+.5f}±{out["sd"][key][n]:.5f})') for n in names]; print('%-34s'%key+f' n={r[ref]["n"]} sh={r[ref]["share"]:.3f}',' '.join('%24s'%x for x in row))
