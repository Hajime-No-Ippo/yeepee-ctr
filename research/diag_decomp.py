# Read-only diagnostic: where is the AUC headroom? (event-has-click vs which-item). Re-predicts with saved v3 models.
import numpy as np, polars as pl, lightgbm as lgb
from sklearn.metrics import roc_auc_score as A
D='/workspace/huawei/v3'
for nm,cut in (('full25',25),('full26',26)):
    m=lgb.Booster(model_file=f'{D}/out/{nm}.txt'); fs=m.feature_name()
    va=pl.read_parquet(f'{D}/feat/val{cut}.parquet')
    p=m.predict(va.select(fs).to_numpy().astype(np.float32))
    va=va.select('eid','clicked','us_cold','us_nohist').with_columns(pl.Series('p',p))
    y=va['clicked'].to_numpy()
    ev=va.group_by('eid').agg(pl.col('clicked').max().alias('y'),pl.col('p').sum().alias('ps'),pl.col('p').max().alias('pm'))
    clk=va.join(ev.filter(pl.col('y')==1).select('eid'),on='eid',how='semi')
    r=clk.with_columns(pl.col('p').rank('average').over('eid').alias('rk')).filter(pl.col('clicked')==1)['rk'].to_numpy()
    j=va.join(ev.select('eid','y','ps'),on='eid',how='left')
    po=np.where(j['y'].to_numpy()==1,p,0)
    # perfect within-event ranking, model event-level score
    pw=j['ps'].to_numpy()+y*100
    # row prob = P(event click)*softmax-share  -> how much does event-level score alone give
    print(nm,'global',round(A(y,p),4),'| event_has_click AUC sum_p',round(A(ev['y'],ev['ps']),4),'max_p',round(A(ev['y'],ev['pm']),4),
          '| within-clicked-event AUC',round(((r-1)/9).mean(),4),'top1',round((r==10).mean(),4),
          '| oracle event-label',round(A(y,po),4),'| global AUC of event-score only',round(A(y,j['ps'].to_numpy()),4))
