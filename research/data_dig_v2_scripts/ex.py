import numpy as np, polars as pl, sys; sys.path.insert(0,'/tmp/dig')
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from h import wauc
T=np.load('tr_ev.npz'); V=np.load('val_t.npz'); Z=np.load('tt_nbnn.npz')
D=pl.read_parquet('/tmp/dig/D.parquet',columns=['eid','clicked','cut'])
E=pl.read_parquet('/tmp/dig/events.parquet').select('eid','tb')
s=Z['dot']; cut=D['cut'].to_numpy(); y=D['clicked'].to_numpy()
G=V['evix']*2+(cut==26); gu,gk=np.unique(G,return_inverse=True)
hl=(V['H']>=0).sum(1); unorm=np.linalg.norm(Z['u'],axis=1)
d=D.with_columns(pl.Series('g',gk),pl.Series('s',s),pl.Series('hl',hl[V['evix']]),pl.Series('un',unorm[V['evix']])).join(E,on='eid',how='left')
# 1(c) calibration of within-event softmax (clicked events only), temperature fitted by 1-D grid
dc=d.filter(pl.col('clicked').max().over('g')==1)
best=None
for tau in np.linspace(0.2,3,29):
    p=dc.with_columns((pl.col('s')*tau).exp().alias('e')).with_columns((pl.col('e')/pl.col('e').sum().over('g')).alias('p'))
    ll=-np.log(p.filter(pl.col('clicked')==1)['p'].to_numpy()).mean()
    if best is None or ll<best[0]: best=(ll,tau,p)
ll,tau,p=best; print('softmax temperature',round(tau,2),'NLL/clicked event',round(ll,4),'uniform',round(np.log(10),4))
p=p.with_columns(pl.col('p').qcut(10,labels=[str(i) for i in range(10)]).alias('bin'))
print(p.group_by('bin').agg(pl.col('p').mean().round(3).alias('pred'),pl.col('clicked').mean().round(3).alias('obs'),pl.len()).sort('bin'))
# taste strength vs history length; within-AUC by history length and time bucket
for col,br in (('hl',[0,5,10,15,19,20]),('tb',None)):
    dd=d.with_columns(pl.col(col).cut(br).alias('b') if br else pl.col(col).alias('b'))
    out=[]
    for b in sorted(dd['b'].unique().to_list(),key=str):
        x=dd.filter(pl.col('b')==b); 
        if x['g'].n_unique()>100: out.append((str(b),x['g'].n_unique(),round(wauc(x.rename({'g':'_g'}).with_columns(pl.col('_g').alias('eid')),'s'),4),round(x['un'].mean(),3)))
    print(col,'(bucket, events, withinAUC tt, mean|u|)',out)
# event level: is the slate gain user-level (|u|) or slate-specific?
uc=pl.DataFrame({'user':T['user'],'c':(T['lab']<10).astype(float)}).group_by('user').agg(pl.col('c').sum().alias('cs'),pl.len().alias('n'))
ev=d.group_by('g').agg(pl.col('clicked').max().alias('y'),pl.col('cut').first(),pl.col('s').exp().sum().log().alias('lse'),pl.col('s').mean().alias('mn'),pl.col('un').first(),pl.col('hl').first(),pl.col('tb').first()).sort('g')
ev=ev.with_columns(pl.Series('user',V['evuser'][gu//2])).join(uc,on='user',how='left',maintain_order='left').with_columns(((pl.col('cs').fill_null(0)+0.63*5)/(pl.col('n').fill_null(0)+5)).alias('uctr'))
ev=ev.with_columns((pl.col('uctr')/(1-pl.col('uctr'))).log().alias('lu'))
yv=ev['y'].to_numpy(); cv=ev['cut'].to_numpy()
for cols in (['lu'],['lu','un','hl'],['lu','lse','mn'],['lu','un','hl','lse','mn']):
    X=ev.select(cols).to_numpy(); r=[]
    for c in (25,26):
        idx=np.where(cv==c)[0]; f=idx%2; pp=np.zeros(len(idx))
        for ff in (0,1):
            pp[f==ff]=LogisticRegression().fit(X[idx[f!=ff]],yv[idx[f!=ff]]).predict_proba(X[idx[f==ff]])[:,1]
        r.append(round(roc_auc_score(yv[idx],pp),4))
    print('event AUC',cols,r)
