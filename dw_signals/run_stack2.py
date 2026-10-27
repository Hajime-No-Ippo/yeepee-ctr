"""residual-boosting incremental test (init_score = v11 logit). usage: run_stack2.py CUT [groups|all]"""
import sys, json, polars as pl, numpy as np, warnings, os; warnings.filterwarnings('ignore')
from stack import *
c=int(sys.argv[1]); groups_arg=sys.argv[2].split(',') if len(sys.argv)>2 else ['all']
D=Data(); V=D.val(c); y=V['y']; sg=segs(c); user=V['user']; ev=np.repeat(np.arange(len(y)//10),10); stale=sg['stale']; nh=sg['nohist']
Fb=base_frame(D,V,c); init=Fb['v11'].astype(np.float64); Fb2={k:v for k,v in Fb.items() if k!='v11'}
Fs=pl.read_parquet(f'cache/sig_C{c}.parquet')
if os.path.exists(f'cache/extra_C{c}.parquet'):
    Fe=pl.read_parquet(f'cache/extra_C{c}.parquet'); Fs=pl.concat([Fs,Fe],how='horizontal')
Fs=pl.concat([Fs,pl.read_parquet(f'cache/sig2_C{c}.parquet')],how='horizontal') if os.path.exists(f'cache/sig2_C{c}.parquet') else Fs
if os.path.exists(f'cache/sig3_C{c}.parquet'): Fs=pl.concat([Fs,pl.read_parquet(f'cache/sig3_C{c}.parquet')],how='horizontal')
GR=json.load(open('groups.json'))
GR={g:[x for x in cols if x in Fs.columns] for g,cols in GR.items()}
s2=set(pl.read_parquet(f'cache/sig2_C{c}.parquet').columns)|set(pl.read_parquet(f'cache/sig3_C{c}.parquet').columns)
GR['ALL']=[x for x in Fs.columns if not x.startswith('decoy') and x not in s2]
GR['ALL2']=[x for x in Fs.columns if not x.startswith('decoy') and x not in set(pl.read_parquet(f'cache/sig3_C{c}.parquet').columns)]
GR['ALL3']=[x for x in Fs.columns if not x.startswith('decoy')]
seeds=(0,1)
Xb=np.column_stack(list(Fb2.values())).astype(np.float32)
p0=np.mean([oof_init(Xb,y,user,init,s) for s in seeds],0); print('C',c,'v11 raw',A(y,init),'resid-base OOF',A(y,p0),flush=True)
for g,cols in GR.items():
    if not cols: continue
    if groups_arg!=['all'] and g not in groups_arg: continue
    Xn=np.column_stack([Xb]+[Fs[k].to_numpy() for k in cols]).astype(np.float32)
    p1=np.mean([oof_init(Xn,y,user,init,s) for s in seeds],0)
    r=dict(cut=c,method='resid',group=g,v11=A(y,init),base=A(y,p0),new=A(y,p1),delta=A(y,p1)-A(y,p0),delta_vs_v11=A(y,p1)-A(y,init),ci=list(boot(y,p0,p1,ev,60)),feats=len(cols))
    for nm,m in (('stale24p',stale>=24),('stale48p',stale>=48),('nohist',nh==1),('hist',nh==0)):
        if m.sum()>3000: r['d_'+nm]=A(y[m],p1[m])-A(y[m],p0[m]); r['n_'+nm]=int(m.sum())
    print(json.dumps({k:(round(v,5) if isinstance(v,float) else v) for k,v in r.items()}),flush=True); open('stack_results.jsonl','a').write(json.dumps(r)+'\n')
