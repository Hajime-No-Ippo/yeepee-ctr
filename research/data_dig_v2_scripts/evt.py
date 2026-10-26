# event-level: does slate affinity (max / logsumexp of u.v) predict whether the event gets a click, beyond user CTR?
import numpy as np, polars as pl
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
T=np.load('tr_ev.npz'); V=np.load('val_t.npz')
D=pl.read_parquet('/tmp/dig/D.parquet',columns=['eid','clicked','cut'])
uc=pl.DataFrame({'user':T['user'],'c':(T['lab']<10).astype(float)}).group_by('user').agg(pl.col('c').sum().alias('cs'),pl.len().alias('n'))
G=V['evix']*2+(D['cut'].to_numpy()==26)
gu,gk=np.unique(G,return_inverse=True)
ev=D.with_columns(pl.Series('g',gk)).group_by('g').agg(pl.col('clicked').max().alias('y'),pl.col('cut').first()).sort('g')
ev=ev.with_columns(pl.Series('user',V['evuser'][gu//2])).join(uc,on='user',how='left').with_columns(((pl.col('cs').fill_null(0)+0.63*5)/(pl.col('n').fill_null(0)+5)).alias('uctr'))
y=ev['y'].to_numpy(); cut=ev['cut'].to_numpy()
for tag in ('nbnn','nb'):
    Z=np.load(f'tt_{tag}.npz'); s=Z['dot']; K=len(y)
    S=np.full((K,10),-np.inf); 
    order=np.argsort(gk,kind='stable'); kk=gk[order]; pos=np.arange(len(kk))-np.searchsorted(kk,kk)
    S[kk,pos]=s[order]
    lse=np.log(np.exp(S-S.max(1,keepdims=True)).sum(1))+S.max(1); mx=S.max(1); mean=np.nanmean(np.where(np.isinf(S),np.nan,S),1)
    null=Z['null'][gu//2]; pclick=1/(1+np.exp(null-lse))
    feats={'uctr':ev['uctr'].to_numpy(),'lse':lse,'max':mx,'mean':mean,'model_pclick(lse-null)':pclick}
    for c in (25,26):
        m=cut==c; print(tag,c,{k:round(roc_auc_score(y[m],v[m]),4) for k,v in feats.items()})
    # does lse add to uctr? 2-fold CV logistic on events (by k parity)
    X=np.c_[np.log(feats['uctr']/(1-feats['uctr'])),lse,mx,mean]
    for c in (25,26):
        m=cut==c; idx=np.where(m)[0]; f=idx%2
        for cols,nm in (([0],'uctr'),([0,1,2,3],'uctr+lse+max+mean')):
            p=np.zeros(len(idx))
            for ff in (0,1):
                lr=LogisticRegression(C=1.0).fit(X[idx[f!=ff]][:,cols],y[idx[f!=ff]]); p[f==ff]=lr.predict_proba(X[idx[f==ff]][:,cols])[:,1]
            print(tag,c,nm,'event AUC',round(roc_auc_score(y[idx],p),4))
