import sys, json, numpy as np
sys.path.insert(0,'/workspace/huawei/dw_levers')
from choice import *
from common import meta, z12, A
cut=int(sys.argv[1]); cfg=json.loads(sys.argv[2]); tag=sys.argv[3]
D=Data(); v=D.val(cut); m_=meta(cut); assert (m_['clicked'].to_numpy()==v['y']).all()
v['col']=align(D,v); z=z12(cut); print('v12',A(v['y'],z),flush=True)
mdl=fit_choice(D,Jd(cut),cfg,val=v)
lp,on=score(mdl,D,v['e'][::10]); lp=to_val(lp,v['col']); y=v['y']
print('choice AUC',A(y,lp.ravel()),'within',A(y,lp.ravel()-np.repeat(0,len(y))),flush=True)
np.save(f'/workspace/huawei/dw_levers/cache/choice_{tag}_c{cut}.npy',np.column_stack([lp.ravel(),np.repeat(on,10)]).astype(np.float32))
for w in (0.1,0.2,0.3,0.5): print('blend z+w*lp',w,A(y,z+w*lp.ravel()),flush=True)
