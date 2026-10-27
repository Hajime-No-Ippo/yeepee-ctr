"""cmp11.py NAME SPEC [--sqw w --sq DIR]: SPEC = items joined by '+', item = tag:col[:seeds] (seeds comma list, default 0) -> mean raw score.
Reports AUC/rw/buckets per cut (25,23,26) and deltas vs v8 trim (w .2 sq8) and v10B. Appends to cmp11_results.jsonl."""
import sys, json, numpy as np
sys.argv=[sys.argv[0]]+sys.argv[1:]
import cmp10 as C
from sklearn.metrics import roc_auc_score as A
av=sys.argv[1:]
def opt(k,d):
    if k in av: i=av.index(k); v=av[i+1]; del av[i:i+2]; return v
    return d
sqw=float(opt('--sqw','0')); name,spec=av[0],av[1]
def load(item,c):
    p=item.split(':'); tag,col=p[0],int(p[1]); seeds=p[2].split(',') if len(p)>2 else ['0']
    return np.mean([np.load(f'ens/{tag}_c{c}_s{s}.npy')[:,col] for s in seeds],0)
def zspec(spec,c): return np.mean([load(i,c) for i in spec.split('+')],0)
def ref10(c):
    b=np.load(f'/workspace/huawei/v10/ens/v10B_b_c{c}_s0.npy')[:,1]; s=np.load(f'/workspace/huawei/v10/ens/v10B_s_c{c}_s0.npy')[:,2]; return (b+s)/2
for c in (25,23,26):
    try: z=zspec(spec,c)
    except FileNotFoundError: continue
    if sqw: z=(1-sqw)*z+sqw*C.SQ[c]
    s=C.stats(z,c); s8=C.stats(C.z('trim',c,0.2,C.SQ8) if False else (0.8*(np.load(f'/workspace/huawei/v10/ens/trim_b_c{c}_s0.npy')[:,1]+np.load(f'/workspace/huawei/v10/ens/trim_s_c{c}_s0.npy')[:,2])/2+0.2*C.SQ8[c]),c); s10=C.stats(ref10(c),c)
    r=dict(name=name,spec=spec,cut=c,**{k:round(v,5) for k,v in s.items()},**{'d8_'+k:round(s[k]-s8[k],5) for k in ('auc','rw')},**{'d10_'+k:round(s[k]-s10[k],5) for k in s})
    print(json.dumps(r),flush=True); open('cmp11_results.jsonl','a').write(json.dumps(r)+'\n')
