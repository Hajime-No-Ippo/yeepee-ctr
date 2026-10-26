exec(open('svdvar6.py').read().split('for N in')[0])
xs={}
for N,kk in ((30,12),(40,12),(30,13)):
    dfv=A.filter(pl.col('L')-pl.col('p')<=N).unique(['u','ix'])
    s,vt=build(dfv['u'].to_numpy(),dfv['ix'].to_numpy(),LH.height,0.5,kk)
    x,_=cosf(vt[1:kk].T*s[1:kk]); xs[f'l{N}k{kk}']=x
    dd=D.with_columns(pl.Series('x',x))
    print(N,kk,'glob',{c:round(roc_auc_score(y[cut==c],x[cut==c]),4) for c in (25,26)},'within',{c:round(wauc(dd.filter(pl.col('cut')==c),'x'),4) for c in (25,26)},'resid',resid(x,3),flush=True)
    np.save(f'x_l{N}k{kk}.npy',x)
# nearest existing
import json; old=list(json.load(open('/tmp/imp_s0.json'))); Xo=D.select(old).to_pandas().rank()
import pandas as pd
for n,x in xs.items():
    cr=Xo.corrwith(pd.Series(x).rank()).abs().sort_values(ascending=False); print(n,'nearest',list(zip(cr.index[:3],cr.values[:3].round(3))))
# multi-feature residual: add z-scored within event + raw
D2=D.with_columns(pl.Series('x',xs['l40k12'])).with_columns(((pl.col('x')-pl.col('x').mean().over('eid'))/(pl.col('x').std().over('eid')+1e-9)).alias('xz'),(pl.col('x')-pl.col('x').max().over('eid')).alias('xd'))
X3=D2.select('x','xz','xd').to_numpy()
out={25:[],26:[]}
for sd in range(3):
    p=np.zeros(len(y))
    for f in range(5):
        tr=fold!=f; te=~tr
        m=lgb.train(dict(objective='binary',learning_rate=0.03,num_leaves=7,min_data_in_leaf=300,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10,verbose=-1,num_threads=2,seed=sd),lgb.Dataset(X3[tr],y[tr],init_score=base[tr]),150)
        p[te]=base[te]+m.predict(X3[te],raw_score=True)
    for c in (25,26): out[c].append(roc_auc_score(y[cut==c],p[cut==c]))
print('l40k12 raw+z+dmax resid',{c:round(np.mean(v)-roc_auc_score(y[cut==c],base[cut==c]),5) for c,v in out.items()})
