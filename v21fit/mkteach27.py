import os,glob,numpy as np,polars as pl
from scipy.special import expit
from scipy.optimize import brentq
from scipy.stats import rankdata
H='/workspace/huawei'; os.chdir(f'{H}/v21seq')
rid=pl.read_parquet(f'{H}/v12/feat_v3/eval.parquet',columns=['rowid'])['rowid'].to_list(); mrow=pl.read_parquet(f'{H}/erbrown/merged.parquet',columns=['rowid'])['rowid'].to_list()
EN=np.load(f'{H}/erbrown/w10an/eb_new_rank.npy')
def tologit(lp): lp=np.minimum(lp,-1e-6); return lp-np.log1p(-np.exp(lp))
v='A'
L_=lambda f,col:np.load(f'{H}/v19/ens/{f}.npy')[:,col]
b=L_(f'nm{v}s_c27_s0',0); d=L_(f'nm{v}s_s_c27_s0',1); ml=np.mean([L_(f'nm{v}smlp_c27_s{k}',3) for k in (0,1,2)],0); z=(b+d+2*ml)/4
fs=sorted(glob.glob(f'ens/ctl6_{v}_c27_s*.npy'))
M=np.mean([tologit(np.load(f)[:,-1]) for f in fs],0); T=.4*z+.6*M
ENr=pl.DataFrame({'rowid':mrow,'e':EN}).join(pl.DataFrame({'rowid':rid}),on='rowid',how='right',maintain_order='right')['e'].to_numpy()
assert np.isfinite(ENr).all() and len(ENr)==len(T)==529390
rk=rankdata(ENr,method='ordinal').astype(int)-1; sv=np.sort(T); lg=sv[rk]
bb=brentq(lambda b:expit(lg+b).mean()-0.0632,-6,6); p=expit(lg+bb).astype(np.float32)
print('ours n',len(fs),'shift',bb,'mean p',p.mean(),'slate sum max',p.reshape(-1,10).sum(1).max())
from scipy.stats import spearmanr; print('rho EN,T',spearmanr(ENr[::5],T[::5])[0])
np.save('/workspace/huawei/v21fit/teach_EN_c27.npy',p)
