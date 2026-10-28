"""Small leak-safety / reproducibility test: features for val25 recomputed vs cache; and truncating train events after X must not change features."""
import sys; sys.dont_write_bytecode=True
sys.path.insert(0,'/workspace/huawei/dw_v12sig')
from sigs2 import *
D=Data(); V=D.val(25); X=Jd(25)
df=pl.DataFrame({k:np.asarray(v,np.float32) for k,v in taste_metric(D,X,V).items()})
ref=pl.read_parquet('/workspace/huawei/dw_v12sig/cache/tm_c25.parquet')
bad=[(k,float(np.nanmax(np.abs(df[k].to_numpy()-ref[k].to_numpy())))) for k in df.columns if np.nanmax(np.abs(df[k].to_numpy()-ref[k].to_numpy()))>1e-3]
print('tm mismatches vs cache (torch/svd tolerance 1e-3):',bad)
# leak test: change labels of all events with ts>=X -> features must be identical
Y0=D.Y.copy(); nX=int(np.searchsorted(D.ETS,X)); D.Y[nX:]=1-D.Y[nX:]
f1=item_strength(D,X,V); D.Y=Y0
f0=item_strength(D,X,V); assert all(np.array_equal(f0[k],f1[k],equal_nan=True) for k in f0); print('item_strength unchanged when post-cutoff labels flipped: OK')
