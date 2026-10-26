# fixes: swap-oracle for within-event, eval dow/stale/lastact mix, eval-mix-weighted val AUC
import sys, json, numpy as np, polars as pl
sys.path.insert(0,'/workspace/huawei/research/v7_err_scripts'); sys.argv=['x']
exec(open('/workspace/huawei/research/v7_err_scripts/err.py').read().split('ev=frame')[0])
ev=frame('eval')
for s in ('dow','stale','lastact','user'): print('eval',s,ev.group_by(s).len().with_columns((pl.col('len')/ev.height).round(4)).sort(s).rows())
W={}
for c in (25,26):
    d=frame(f'val{c}'); p=((L('v7base',c,0,1)+L('v7base',c,1,1))/2+L('v7slow',c,0,2))/2
    d=d.with_columns(pl.Series('p',p),pl.arange(0,d.height).alias('i')); y=d['clicked'].to_numpy()
    d=d.with_columns(pl.col('clicked').max().over('eid').alias('ey'),pl.col('p').max().over('eid').alias('pm'))
    # within oracle: in clicked events swap clicked row's score with event max
    q=d.with_columns(pl.when((pl.col('ey')==1)&(pl.col('clicked')==1)).then(pl.col('pm')).otherwise(pl.col('p')).alias('pw'))
    am=q.filter(pl.col('ey')==1).with_columns(pl.col('p').rank('ordinal',descending=True).over('eid').alias('r'))
    cp=am.filter(pl.col('clicked')==1).select('eid',pl.col('p').alias('pc'))
    q=q.join(cp,on='eid',how='left').with_columns(pl.when((pl.col('clicked')==0)&(pl.col('p')==pl.col('pm'))&(pl.col('ey')==1)).then(pl.col('pc')).otherwise(pl.col('pw')).alias('pw')).sort('i')
    tot=A(y,p); ow=A(y,q['pw'].to_numpy())
    ob=A(y,np.where(d['ey'].to_numpy()==1,p,-1e3))
    # eval-mix weights over (stale, lastact, user)
    r={}
    for s in ('stale','lastact','user'):
        vs=d.group_by(s).len().with_columns((pl.col('len')/d.height).alias('vs'))
        es=ev.group_by(s).len().with_columns((pl.col('len')/ev.height).alias('es'))
        w=d.select(s).join(vs.join(es,on=s,how='left'),on=s,how='left').with_columns((pl.col('es').fill_null(0)/pl.col('vs')).alias('w'))['w'].to_numpy()
        r[s]=round(A(y,p,sample_weight=w),5)
    W[c]=dict(auc=round(tot,5),oracle_within=round(ow,5),gap_within=round(ow-tot,5),oracle_between=round(ob,5),gap_between=round(ob-tot,5),evalmix_auc=r)
    print(c,W[c])
json.dump(W,open('/workspace/huawei/research/v7_err_scripts/err2.json','w'),indent=1)
