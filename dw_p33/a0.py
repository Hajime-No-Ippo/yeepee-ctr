import pandas as pd,numpy as np,zipfile,io
from scipy.stats import spearmanr,norm
import polars as pl
H='/workspace/huawei/'
p=pd.read_parquet(H+'teammate/work33/p33_recovered_ranks.parquet')
v8=pd.read_parquet(H+'teammate/work/v8.parquet'); rg=pd.read_csv(H+'v12x/v8regen/submission_v8_regen.csv')
v12=pd.read_csv(H+'v12/out/submission_v12.csv')
z=zipfile.ZipFile(H+'teammate/submission_p33_blend_w70.zip'); b=pd.read_csv(io.BytesIO(z.read(z.namelist()[0])))
N=len(p); rk=lambda x: pd.Series(x).rank(pct=False).to_numpy()
r8=rk(v8.predicted_probability if 'predicted_probability' in v8 else v8.iloc[:,1]); 
r8=rk(v8.predicted_probability.values); r12=rk(v12.predicted_probability.values); rr=rk(rg.predicted_probability.values); k=p.p33_rank.values
print('N',N)
# recipe check
bb=(0.7*r8/N+0.3*k/N); print('recipe v8 file maxdiff',np.abs(bb-b.iloc[:,1].values).max())
print('regen v8 recipe maxdiff',np.abs(0.7*rr/N+0.3*k/N-b.iloc[:,1].values).max(), 'sp(v8,regen)',spearmanr(r8,rr)[0])
S=lambda a,c: spearmanr(a,c)[0]
print('sp p33~v8',S(k,r8),'p33~v12',S(k,r12),'v8~v12',S(r8,r12))
ns=lambda r: norm.ppf((r-0.5)/N)
a,c,d=ns(k),ns(r8),ns(r12)
X=np.c_[c,np.ones(N)]; co=np.linalg.lstsq(X,a,rcond=None)[0]; print('p33_ns = %.3f v8_ns + %.3f ; R2 %.4f'%(co[0],co[1],1-((a-X@co).var()/a.var())))
X=np.c_[d,np.ones(N)]; co=np.linalg.lstsq(X,a,rcond=None)[0]; print('p33_ns ~ v12_ns: %.3f R2 %.4f'%(co[0],1-((a-X@co).var()/a.var())))
X=np.c_[c,d,np.ones(N)]; co=np.linalg.lstsq(X,a,rcond=None)[0]; print('p33_ns ~ v8+v12',co,'R2',1-((a-X@co).var()/a.var()))
# hypothesis: p33 = mix w*v8 + (1-w)*own, own independent of v8 given v12-like?  Test: if p33 contained 0.75 v8, corr with v8 would be >= .75*... use cold events
m=pl.read_parquet(H+'dw_v12sig/t1_resid.parquet'); print(m.columns[:30], m.shape)
import json
ev=pl.read_csv(H+'data/evaluation.csv',columns=['rowid','event_id','history_item_ids','item_id','timestamp'])
hl=ev['history_item_ids'].cast(pl.Utf8).str.count_matches(',')+1
cold=((ev['history_item_ids'].cast(pl.Utf8).str.strip_chars()=='[]')|ev['history_item_ids'].is_null()).to_numpy()
print('cold rows',cold.sum())
def wd(x): m=x.reshape(-1,10); return (m-m.mean(1,keepdims=True)).ravel()
# events are contiguous? check
eid=ev['event_id'].to_numpy(); print('contiguous',(eid.reshape(-1,10)==eid.reshape(-1,10)[:,:1]).all())
a_=wd(a); c_=wd(c); d_=wd(d)
for nm,msk in (('cold',cold),('warm',~cold)):
    mm=msk.reshape(-1,10)[:,0]; idx=np.repeat(mm,10)
    for nm2,o in (('v8',c_),('v12',d_)):
        s=(a_[idx]*o[idx]).sum()/(o[idx]**2).sum(); r=np.corrcoef(a_[idx],o[idx])[0,1]
        print(nm,nm2,'within-event slope %.3f corr %.3f'%(s,r))
# popularity proxy: train item impression count
print('---- solo est')
out={'recipe_maxdiff':float(np.abs(bb-b.iloc[:,1].values).max()),'sp_k_v12':S(k,r12),'sp_k_v8':S(k,r8)}
res={}
for aa in (0.0,0.12,0.25):
    s=a-aa*c; s=(s-s.mean())/s.std(); rs=rk(s)
    res[aa]=(S(rs,r12),S(rs,r8)); print('a=%.2f spearman solo vs v12 %.4f vs v8 %.4f'%(aa,*res[aa]))
    if aa==0.12: sel=rs
pd.DataFrame({'rowid':p.rowid,'p33_solo_rank':k,'p33_solo_rank_pct':k/N,'p33_solo_adj_a012_rank':sel,'p33_solo_adj_a012_pct':sel/N}).to_parquet('p33_solo_est.parquet')
