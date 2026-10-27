import sys, json, polars as pl, numpy as np, warnings; warnings.filterwarnings('ignore')
from stack import *
c=int(sys.argv[1]); groups_arg=sys.argv[2].split(',') if len(sys.argv)>2 else None
D=Data(); V=D.val(c); y=V['y']; sg=segs(c); user=V['user']; ev=np.repeat(np.arange(len(y)//10),10); stale=sg['stale']; nh=sg['nohist']
Fb=base_frame(D,V,c); Fs=pl.read_parquet(f'cache/sig_C{c}.parquet')
GR={'skip':['c_skall','c_skrec','c_impall','c_hist','c_hist_minus_sk','c_hist_minus_skrec'],
 'impsvd':[x for x in Fs.columns if x.startswith('impsvd')],
 'central':[x for x in Fs.columns if x.startswith(('cent_','slate_'))],
 'expo':['expo_resid','expo_ratio','skip_cnt','skip_frac'],
 'usertime':['hr_click_match','hr_ev_match','last_ev_clicked','ev_ctr_last5','ev_ctr_samebucket','ev_cnt_samebucket_frac'],
 'coimp':[x for x in Fs.columns if x.startswith('coimp')]}
extra=json.load(open('groups_extra.json')) if __import__('os').path.exists('groups_extra.json') else {}
for k,v in extra.items():
    if all(x in Fs.columns for x in v): GR[k]=v
GR['ALL']=[x for x in Fs.columns]
seeds=(0,1)
Xb=np.column_stack(list(Fb.values())).astype(np.float32)
p0=np.mean([oof(Xb,y,user,s) for s in seeds],0); print('C',c,'base OOF',A(y,p0),'v11 raw',A(y,Fb['v11']),flush=True)
for g,cols in GR.items():
    if groups_arg and g not in groups_arg and groups_arg!=['all']: continue
    Xn=np.column_stack([Xb]+[Fs[k].to_numpy() for k in cols]).astype(np.float32)
    p1=np.mean([oof(Xn,y,user,s) for s in seeds],0)
    r=dict(cut=c,group=g,base=A(y,p0),new=A(y,p1),delta=A(y,p1)-A(y,p0),ci=list(boot(y,p0,p1,ev,60)),feats=len(cols))
    for nm,m in (('stale24p',stale>=24),('stale48p',stale>=48),('nohist',nh==1),('hist',nh==0)):
        if m.sum()>3000: r['d_'+nm]=A(y[m],p1[m])-A(y[m],p0[m]); r['n_'+nm]=int(m.sum())
    print(json.dumps({k:(round(v,5) if isinstance(v,float) else v) for k,v in r.items()}),flush=True); open('stack_results.jsonl','a').write(json.dumps(r)+'\n')
