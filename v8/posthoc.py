import json,glob,os,numpy as np,polars as pl
from sklearn.metrics import roc_auc_score
from cl import CACHE,KEEP
import hashlib
def meta(n):
    key=hashlib.md5(json.dumps([n,1.0,'notime',[],os.path.getmtime(f'/workspace/huawei/v8/feat_v3/{n}.parquet')]).encode()).hexdigest()[:12]
    return pl.read_parquet(f'{CACHE}/{n}_{key}.parquet',columns=['rowid','eid','clicked'])
def evaucs(df,s):
    d=df.with_columns(s=pl.Series(s))
    E=d.group_by('eid').agg(pl.col('clicked').max(),pl.col('s').max())
    W=d.group_by('eid').agg((pl.col('clicked').max()>pl.col('clicked').min()).alias('mx'),pl.col('clicked'),pl.col('s')).filter('mx')
    R=d.with_columns(r=pl.col('s').rank('average').over('eid')).group_by('eid').agg(np_=pl.col('clicked').sum(),n=pl.len(),rs=(pl.col('r')*pl.col('clicked')).sum())
    R=R.filter((pl.col('np_')>0)&(pl.col('np_')<pl.col('n')))
    wa=((R['rs']-R['np_']*(R['np_']+1)/2)/(R['np_']*(R['n']-R['np_']))).mean()
    return dict(glob=round(roc_auc_score(d['clicked'],d['s']),5),event=round(roc_auc_score(E['clicked'],E['s']),5),within=round(wa,5))
out={}
for cut in (25,26):
    m=meta(f'val{cut}'); p=np.load(f'scr_base_c{cut}.npy')
    ev=pl.read_parquet(f'feat_ev/c{cut}/val{cut}.parquet')
    df=m.with_columns(it=pl.Series(p)).join(ev,on='rowid',how='left')
    assert df.height==m.height
    pe=df['ev2_pev'].to_numpy().clip(1e-6,1-1e-6); lpe=np.log(pe/(1-pe)); it=df['it'].to_numpy()
    d=df.with_columns(lse=pl.col('it').exp().sum().over('eid').log(), ec=pl.col('it').max().over('eid'))
    lsm=it-d['lse'].to_numpy()
    r={'item':evaucs(m,it),'pev_only':evaucs(m,lpe),'b_pev_x_softmax':evaucs(m,pe*np.exp(lsm))}
    for al in (0.5,0.75,1.0,1.25,1.5,2.0,3.0):
        r[f'c_a{al}']=evaucs(m,lpe+al*it)
        r[f'c2_logsm_a{al}']=evaucs(m,np.log(pe)+al*lsm)
    out[cut]=r; print(cut,json.dumps(r),flush=True)
json.dump(out,open('posthoc_B3.json','w'),indent=1)
