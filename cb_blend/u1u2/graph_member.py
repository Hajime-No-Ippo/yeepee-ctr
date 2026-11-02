"""Graph member for blend_oof: u1/u1b LightGCN features (v21graph), per-seed caches in cb_blend/u1u2/feat_c{cut}_{tag}_s{seed}.npz
(built by build_members.py from v21graph/out/emb_*.npz; C23 u1b 800 steps k3, C25/C26 u1 350 steps k5/k6). Member = z(demean_slate(mean_seed s_uv)) + z(.. s_h) + z(.. cos_mean)
('dm sum3' as v21graph/ms.py). NB slate demean (rows//10) is a per-slate normalisation; use raw=True for the pure global variant."""
import glob,numpy as np
D='/workspace/huawei/cb_blend/u1u2'; TAG={23:'u1b',25:'u1',26:'u1'}
def graph_member(c,demean=True):
    fs=glob.glob(f'{D}/feat_c{c}_{TAG[c]}_s*.npz'); assert fs,c
    zs=lambda x:(x-x.mean())/x.std(); dm=lambda s:(s.reshape(-1,10)-s.reshape(-1,10).mean(1,keepdims=True)).ravel()
    f=dm if demean else (lambda x:x); P=[np.load(x) for x in fs]
    return sum(zs(f(np.mean([p[k].astype(np.float64) for p in P],0))) for k in ('s_uv','s_h','cos_mean'))
