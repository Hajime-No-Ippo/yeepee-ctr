import numpy as np, polars as pl, zipfile, io
from scipy.stats import rankdata, spearmanr
H='/workspace/huawei'
D=pl.read_parquet('../merged.parquet'); n=D.height
def rd(zp):
    z=zipfile.ZipFile(zp); return pl.read_csv(io.BytesIO(z.read(z.namelist()[0])))
w=rd('submission_blend_newbest_v21_w10_20261031.csv.zip')
print('same order',w['rowid'].to_list()==D['rowid'].to_list(), w.shape)
W=w['predicted_probability'].to_numpy(); print('w10 min max mean uniq',W.min(),W.max(),W.mean(),len(np.unique(W)))
cands={'ours(v21listB)':D['ours'].to_numpy(),'pr(p33_w80)':D['pr'].to_numpy(),'p33':D['p33_rank'].to_numpy(),'oldEB':D['eb'].to_numpy()}
for s in ['v21slate_B_TRANSDUCTIVE','v21slate_A_trainonly','v21list_A_trainonly']:
    x=rd(f'{H}/v21diag/out/submission_{s}.csv.zip'); x=D.select('rowid').join(x,on='rowid',how='left'); cands[s]=x['predicted_probability'].to_numpy()
nr=lambda x:(rankdata(x)-1)/(n-1)
print('old EB stats',D['eb'].min(),D['eb'].max(),D['eb'].mean(),'ours',D['ours'].min(),D['ours'].max(),D['ours'].mean())
Wr=nr(W)
for k,v in cands.items():
    print(k,'spearman w10',round(spearmanr(W,v)[0],5),'pearson prob',round(np.corrcoef(W,v)[0,1],5))
# is w10 values distributed like rank in [0,1]? 
print('w10 uniformity: pct',np.percentile(W,[1,10,50,90,99]))
# Prob-space: W ~ a*ours + b*Y -> look at Y=W-0.1*ours
for k in ['ours(v21listB)','pr(p33_w80)','v21slate_B_TRANSDUCTIVE','v21slate_A_trainonly','v21list_A_trainonly']:
    X=cands[k]
    for sp,(wv,xv) in {'prob':(W,X),'rank':(Wr,nr(X))}.items():
        for a in (0.1,):
            Y=(wv-a*xv)/(1-a)
            print(k,sp,'recovered Y min/max',round(Y.min(),4),round(Y.max(),4),'frac<0',(Y<0).mean().round(4),'frac>1',(Y>1).mean().round(4))
