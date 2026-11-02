import sys; sys.dont_write_bytecode=True
sys.path.insert(0,'/workspace/huawei/v21graph'); sys.path.insert(0,'/workspace/huawei/cb_blend/u1u2')
import numpy as np, gfeat, gfeat_chunk as gc
T=gc.load_tab('val25'); E=2500
S={k:(v[:E*10] if k in('rowid','user','item','y') else v[:E]) for k,v in T.items()}
gfeat.load_tab=lambda nm:S
e=np.load(gc.O+'emb_u1_k5_s1.npz'); U,V=e['U350'],e['V350']
a=gfeat.feats('x',U,V); b=gc.feats_chunk(S,U,V,ch=700)
for k in ('s_uv','s_h','cos_mean'): print(k,float(np.abs(a[k]-b[k]).max()),float(np.abs(a[k]).mean()))
