import sys; sys.dont_write_bytecode=True
sys.path.insert(0,'/workspace/huawei/dw_p33')
from p33feat import *
D=Data(); V=D.val(25); X=Jd(25)
ref=pl.read_parquet('cache/pp_c25.parquet')
f0={k:np.asarray(v) for k,v in pp_feats(D,X,V,groups=('cos','prod','user')).items()}
bad=[(k,float(np.nanmax(np.abs(f0[k]-ref[k].to_numpy())))) for k in f0 if np.nanmax(np.abs(f0[k]-ref[k].to_numpy()))>1e-3]
print('mismatch vs cache (>1e-3):',bad)
Y0=D.Y.copy(); nX=int(np.searchsorted(D.ETS,X)); D.Y[nX:]=1-D.Y[nX:]
f1={k:np.asarray(v) for k,v in pp_feats(D,X,V,groups=('cos','prod','user')).items()}; D.Y=Y0
print('flip post-cutoff labels -> identical:',all(np.array_equal(f0[k],f1[k],equal_nan=True) for k in f0))
