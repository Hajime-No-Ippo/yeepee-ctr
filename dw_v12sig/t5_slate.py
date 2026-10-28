import numpy as np, polars as pl
H='/workspace/huawei'
for nm in ['train','evaluation']:
    d=pl.read_csv(f'{H}/data/{nm}.csv',columns=['rowid','event_id','item_id']+(['clicked'] if nm=='train' else []))
    n=d.height; ev=d['event_id'].to_numpy()
    contiguous=(ev.reshape(-1,10)==ev.reshape(-1,10)[:,:1]).all(); print(nm,n,'events contiguous 10-blocks',contiguous)
    pos=np.tile(np.arange(10),n//10)
    if nm=='train':
        y=d['clicked'].to_numpy(); print('click rate by file position',np.round([y[pos==k].mean() for k in range(10)],4))
        # chi2 test
        from scipy.stats import chisquare
        c=np.array([y[pos==k].sum() for k in range(10)]); print('chi2 p',chisquare(c).pvalue)
        # time-split: is first-half vs second-half same
        # rowid hash order within event? sorted by rowid?
        rid=d['rowid'].to_numpy().reshape(-1,10); print('rowid sorted within event frac',(np.diff(rid.astype('U70').view('U70') if False else np.argsort(rid,1),axis=1)!=1).any(1).mean())
        it=d['item_id'].to_numpy().reshape(-1,10); print('item sorted within event frac',np.mean([ (it[i]==np.sort(it[i])).all() for i in range(20000)]))
        # position by item popularity
        pop=d.group_by('item_id').len().rename({'len':'pop'}); dd=d.with_columns(pl.Series('pos',pos)).join(pop,on='item_id')
        print(dd.group_by('pos').agg(pl.col('pop').mean()).sort('pos')['pop'].to_numpy().round(1))
    else:
        it=d['item_id'].to_numpy().reshape(-1,10); print('item sorted within event frac',np.mean([ (it[i]==np.sort(it[i])).all() for i in range(20000)]))
