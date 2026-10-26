import numpy as np, polars as pl, lightgbm as lgb, sys
sys.path.insert(0,'/workspace/huawei/v2work'); sys.path.insert(1,'/workspace/huawei/v3')
from sklearn.metrics import roc_auc_score
P='/workspace/huawei/v2work/feat_v3'
cols=['rowid','eid','us_cold','us_nohist','tm_stale','ts']
va=pl.read_parquet(f'{P}/val25.parquet',columns=cols+['clicked']); ev=pl.read_parquet(f'{P}/eval.parquet',columns=cols)
p=np.load('combo_val25_notime.npy'); y=va['clicked'].to_numpy()
def seg(d):
    c=d['us_cold'].to_numpy()==1; n=d['us_nohist'].to_numpy()==1; s=d['tm_stale'].to_numpy()>=86400
    return {'warm':~c,'history_only':c&~n,'empty':n,'stale_ge1d':s,'stale_lt1d':~s}
sv=seg(va); se=seg(ev)
aw=roc_auc_score(y[sv['warm']],p[sv['warm']])
print('seg  val_share eval_share val_auc gap_to_warm impact_eval(share*gap)')
for k in sv:
    a=roc_auc_score(y[sv[k]],p[sv[k]]); print(f"{k:13s} {sv[k].mean():.4f} {se[k].mean():.4f} {a:.4f} {aw-a:+.4f} {se[k].mean()*(aw-a):.4f}")
st=ev['tm_stale'].to_numpy()/86400; print('eval stale days quantiles',np.percentile(st,[10,25,50,75,90]).round(2), 'share>=2d',(st>=2).mean().round(3),'>=3d',(st>=3).mean().round(3))
dow=((ev['ts'].to_numpy()//86400)+3)%7; print('eval dow shares',np.bincount(dow,minlength=7)/len(dow))
day=(ev['ts'].to_numpy()-1737936000)//86400; print('eval day-offset shares',dict(zip(*np.unique(day,return_counts=True))))
