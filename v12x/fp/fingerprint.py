import sys, json, os, numpy as np, pandas as pd, polars as pl
from scipy.stats import rankdata, spearmanr
V='/workspace/huawei/v12/ens'; X='/workspace/huawei/v12x/ens'
m=pl.read_parquet('/workspace/huawei/v12/feat_v3/eval.parquet',columns=['rowid','eid','hist_in']); n=m.height
rid=m['rowid'].to_numpy(); eid=m['eid'].to_numpy(); hin=m['hist_in'].to_numpy().astype(float)
g=eid.reshape(-1,10); assert (g==g[:,:1]).all()
q=pd.read_parquet('/workspace/huawei/teammate/work33/p33_recovered_ranks.parquet').set_index('rowid').p33_rank.loc[rid].values
def ld(p,c): return np.load(p)[:,c].astype(np.float64)
M={}
big=ld(f'{V}/dwbig_c27_s0.npy',1); sm=ld(f'{V}/dw_s_c27_s0.npy',1); ml=np.mean([ld(f'{V}/dwmlp_c27_s%d.npy'%s,3) for s in range(3)],0)
M['v12 blend']=(big+sm+2*ml)/4; M['v12 dwbig (LGB)']=big; M['v12 dw_s (LGB)']=sm; M['v12 MLP mean3']=ml
xg=ld(f'{X}/xgb_c27_s0.npy',1) if os.path.exists(f'{X}/xgb_c27_s0.npy') else None
if xg is not None: M['v12 XGB @450']=xg
v11=os.path.exists('/workspace/huawei/v11/ens/sCbig_c27_s0.npy')
M['v11 sCbig (LGB)']=ld('/workspace/huawei/v11/ens/sCbig_c27_s0.npy',1)
import glob
fs=sorted(glob.glob('/workspace/huawei/v2work/ens/base_c27_s*.npy')); M['v2work base LGB (v3-era, %d seeds)'%len(fs)]=np.mean([np.load(f)[:,1] for f in fs],0)
# v8 regen members
R='/workspace/huawei/v12x/v8regen'
if os.path.exists(R+'/v8_final.npy'):
    d=np.load(R+'/v8_final.npz'); 
def rk(s): return (rankdata(s,'average')-1)/(len(s)-1)
if os.path.exists('/workspace/huawei/v12x/fp/v8_members.npz'):
    z=np.load('/workspace/huawei/v12x/fp/v8_members.npz')
    for k in z.files: M[k]=z[k]
if xg is not None:
    for w in (.3,.5,.7): M[f'rank blend {w:.1f}*XGB+{1-w:.1f}*dwbig']=w*rk(xg)+(1-w)*rk(big)
    for w in (.3,.5,.7): M[f'rank blend {w:.1f}*XGB+{1-w:.1f}*v12blend']=w*rk(xg)+(1-w)*rk(M['v12 blend'])
def wz(s):  # within-event rank z-score
    r=np.argsort(np.argsort(s.reshape(-1,10),1),1).astype(float); return ((r-4.5)/np.sqrt(8.25)).ravel()
def wspear(s,t):
    a=s.reshape(-1,10); b=t.reshape(-1,10)
    ra=np.argsort(np.argsort(a,1),1).astype(float); rb=np.argsort(np.argsort(b,1),1).astype(float)
    ra-=ra.mean(1,keepdims=True); rb-=rb.mean(1,keepdims=True); return float((ra*rb).sum(1).mean()/np.sqrt((ra**2).sum(1).mean()*(rb**2).sum(1).mean()))
def hg(s):  # mean within-event rank-z of in-history candidates minus non-history (events having both)
    z=wz(s); return float(z[hin==1].mean()-z[hin==0].mean())
rows=[]
for k,s in M.items():
    rows.append(dict(model=k,spearman_global=round(spearmanr(s,q)[0],4),spearman_within=round(wspear(s,q),4),inhist_gap_sd=round(hg(s),3)))
rows.append(dict(model='p33 itself',spearman_global=1,spearman_within=1,inhist_gap_sd=round(hg(q),3)))
df=pd.DataFrame(rows); print(df.to_string(index=False)); df.to_csv('/workspace/huawei/v12x/fp/fingerprint.csv',index=False)
print('hist_in share',hin.mean(),'p33-minus-v12 gap', round(hg(q)-hg(M['v12 blend']),3))
from scipy.stats import norm
def ns(s): return norm.ppf((rankdata(s)-.5)/len(s))
zq=ns(q); print('\nDW-style: mean over in-history rows of (normal-score(p33) - normal-score(model)), in sd units; and same restricted to hist_in==0')
out=[]
for k,s in M.items():
    d=zq-ns(s); out.append((k,round(float(d[hin==1].mean()),3),round(float(d[hin==0].mean()),4)))
for o in out: print('%-45s %6.3f %8.4f'%o)
