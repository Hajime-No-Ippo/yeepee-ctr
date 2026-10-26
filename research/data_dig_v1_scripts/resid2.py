import polars as pl, numpy as np, lightgbm as lgb, sys
from sklearn.metrics import roc_auc_score
sys.path.insert(0,'/tmp/dig'); from h import wauc
D=pl.read_parquet('D.parquet')
rng=np.random.default_rng(0)
for k in range(3): D=D.with_columns(pl.Series(f'noise{k}',rng.normal(size=D.height)))
fold=(D['user'].hash(seed=3)%5).to_numpy()
y=D['clicked'].to_numpy(); base=D['base'].to_numpy(); cut=D['cut'].to_numpy()
def run(feats,seeds=3):
    res={25:[],26:[]}
    for s in range(seeds):
        p=np.zeros(len(y))
        for f in range(5):
            tr=fold!=f; te=~tr
            X=D.select(feats).to_numpy()
            ds=lgb.Dataset(X[tr],y[tr],init_score=base[tr])
            m=lgb.train(dict(objective='binary',learning_rate=0.03,num_leaves=7,min_data_in_leaf=300,feature_fraction=0.9,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10,verbose=-1,num_threads=2,seed=s),ds,150)
            p[te]=base[te]+m.predict(X[te],raw_score=True)
        for c in (25,26):
            mk=cut==c; res[c].append(roc_auc_score(y[mk],p[mk]))
    return {c:round(np.mean(v),5) for c,v in res.items()}
b={c:round(roc_auc_score(y[cut==c],base[cut==c]),5) for c in (25,26)}
print('base (sample)',b)
for name,fs in [('control noise',['noise0','noise1','noise2']),('control base-only feat',['base']),
                ('slate_c',['slate_c']),('slate_s',['slate_s']),('uimp_c',['uimp_c']),('uimp_s',['uimp_s']),('uh_s',['uh_s']),('uh_c',['uh_c']),
                ('all slate/imp',['slate_c','slate_s','uimp_c','uimp_s','uh_s','uh_c'])]:
    r=run(fs); print(f'{name:24s}',r,'delta',{c:round(r[c]-b[c],5) for c in r})
