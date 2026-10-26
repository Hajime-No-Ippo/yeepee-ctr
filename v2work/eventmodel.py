"""Step B: event-level model P(event has any click). Leak-free OOF:
setting c25: train/OOF on f1..f4 (4 user-hash groups), predict val25 with model on all f1..f4
setting c26: same on f1..f5 -> val26;  setting final: f1..f6 -> eval.
Writes feat_ev/<setting>/<table>.parquet (rowid, ev2_pev, ev2_pev_lgt) for the main model, and prints event AUCs."""
import sys, os, json, numpy as np, polars as pl, lightgbm as lgb
from sklearn.metrics import roc_auc_score
D='/workspace/huawei/v2work'; J20=1737331200; DAY=86400
def ev_table(n):
    a=pl.read_parquet(f'{D}/feat_v3/{n}.parquet').join(pl.read_parquet(f'{D}/feat_x2/{n}.parquet'),on='rowid',how='left')
    a=a.join(pl.read_parquet(f'{D}/feat_x/{n}.parquet',columns=['rowid','x_con05_k32','x_icon05_k64','x_con05_k64']),on='rowid',how='left')
    U=[c for c in a.columns if c.startswith(('us_','s_'))]+['tm_hour','tm_stale']
    EV=[c for c in a.columns if c.startswith('ev_') and c.endswith(('_max','_mean'))]
    agg=[pl.col(c).first() for c in U+EV+['u','ts']]+[pl.col(c).max().alias(c+'_emax') for c in ('x_con05_k32','x_icon05_k64','x_con05_k64')]+\
        [pl.col(c).std().alias(c+'_estd') for c in ('x_con05_k32','x_icon05_k64')]
    if 'clicked' in a.columns: agg.append(pl.col('clicked').max().alias('y'))
    e=a.group_by('eid').agg(agg)
    nd=((pl.col('ts')-pl.col('tm_stale')-J20)/DAY)
    e=e.with_columns(((pl.col('ts')//DAY+3)%7).alias('dow'),(pl.col('us_nev')/nd).alias('us_nev_pd'),(pl.col('us_nclk_win')/nd).alias('us_nclk_pd'),
                     (pl.col('us_nclk_all')/(nd+7)).alias('us_nclkall_pd'))
    return e
FS={'user_time':lambda e:[c for c in e.columns if c.startswith(('us_','s_'))]+['tm_hour','dow'],
    'user_time_stale':lambda e:[c for c in e.columns if c.startswith(('us_','s_'))]+['tm_hour','dow','tm_stale'],
    'full':lambda e:[c for c in e.columns if c.startswith(('us_','s_','ev_'))or c.endswith(('_emax','_estd'))]+['tm_hour','dow']}
P=dict(objective='binary',learning_rate=0.03,num_leaves=31,min_data_in_leaf=100,feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,
       lambda_l2=10,verbose=-1,num_threads=6)
NR=600
SET={'c25':([f'f{k}' for k in range(1,5)],'val25'),'c26':([f'f{k}' for k in range(1,6)],'val26'),'final':([f'f{k}' for k in range(1,7)],'eval')}
if __name__=='__main__':
    setting=sys.argv[1]; fsn=sys.argv[2] if len(sys.argv)>2 else 'full'; write='--write' in sys.argv
    trn,tst=SET[setting]
    T={n:ev_table(n) for n in trn+[tst]}
    tr=pl.concat([T[n].with_columns(pl.lit(n).alias('tab')) for n in trn],how='diagonal_relaxed')
    fs=FS[fsn](tr); X=tr.select(fs).to_numpy().astype(np.float32); y=tr['y'].to_numpy()
    grp=(tr['u'].hash(seed=3)%4).to_numpy()
    oof=np.zeros(len(y))
    for k in range(4):
        m=lgb.train(P,lgb.Dataset(X[grp!=k],y[grp!=k]),NR); oof[grp==k]=m.predict(X[grp==k])
    m=lgb.train(P,lgb.Dataset(X,y),NR); te=T[tst]; pt=m.predict(te.select(fs).to_numpy().astype(np.float32))
    res=dict(setting=setting,fs=fsn,n_feats=len(fs),oof_auc=round(roc_auc_score(y,oof),5),ev_rate=round(float(y.mean()),4))
    if 'y' in te.columns: res['test_event_auc']=round(roc_auc_score(te['y'].to_numpy(),pt),5)
    g=m.feature_importance('gain'); res['top']=[(fs[i],round(float(g[i]/g.sum()),3)) for i in np.argsort(-g)[:10]]
    print(json.dumps(res),flush=True); open(f'{D}/event_results.jsonl','a').write(json.dumps(res)+'\n')
    if write:
        out=f'{D}/feat_ev/{setting}'; os.makedirs(out,exist_ok=True)
        tr=tr.with_columns(pl.Series('pev',oof)); te=te.with_columns(pl.Series('pev',pt))
        for n in trn+[tst]:
            ee=(tr.filter(pl.col('tab')==n) if n in trn else te).select('eid','pev')
            rows=pl.read_parquet(f'{D}/feat_v3/{n}.parquet',columns=['rowid','eid']).join(ee,on='eid',how='left')
            p=rows['pev'].to_numpy().clip(1e-4,1-1e-4)
            rows.select('rowid').with_columns(pl.Series('ev2_pev',p.astype(np.float32)),pl.Series('ev2_pev_lgt',np.log(p/(1-p)).astype(np.float32))).write_parquet(f'{out}/{n}.parquet')
        print('wrote',out)
