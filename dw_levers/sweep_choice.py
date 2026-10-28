import sys, json, numpy as np
sys.path.insert(0,'/workspace/huawei/dw_levers')
from choice import *
from common import meta, z12, A
cut=int(sys.argv[1]); cfgs=[json.loads(l) for l in open(sys.argv[2]) if l.strip()]
D=Data(); v=D.val(cut); v['col']=align(D,v); z=z12(cut)
for cfg in cfgs:
    print('CFG',json.dumps(cfg),flush=True)
    mdl=fit_choice(D,Jd(cut),cfg,val=v); lp,on=score(mdl,D,v['e'][::10]); lp=to_val(lp,v['col']).ravel(); y=v['y']
    r=dict(cfg=cfg,cut=cut,auc=A(y,lp),blend={w:round(A(y,z+w*lp)-A(y,z),5) for w in (0.1,0.2,0.4)}); print('RES',json.dumps(r),flush=True)
    open('choice_sweep.jsonl','a').write(json.dumps(r)+'\n')
