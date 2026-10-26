import polars as pl, numpy as np
J20=1737331200; DAY=86400; C=J20+5*DAY
imap=pl.read_parquet('/tmp/dig/imap.parquet')
EV=pl.read_parquet('/workspace/huawei/v2work/train_events.parquet',columns=['eid','user','ts','tb','h_items']).filter(pl.col('ts')<C)
R=(pl.scan_parquet('/workspace/huawei/v2work/train_rows.parquet').select('eid','item','clicked').join(EV.lazy().select('eid'),on='eid')
   .join(imap.lazy(),on='item').collect())
G=R.group_by('eid',maintain_order=True).agg('ix','clicked').filter(pl.col('ix').list.len()==10)
G=G.join(EV,on='eid').sort('ts')
cand=np.array(G['ix'].to_list(),np.int32); clk=np.array(G['clicked'].to_list(),np.int8)
lab=np.where(clk.sum(1)>0,clk.argmax(1),10).astype(np.int8)
hh=G.select(pl.col('h_items').list.eval(pl.element().replace_strict(imap['item'],imap['ix'],default=-1)))['h_items'].to_list()
Hm=np.full((len(hh),20),-1,np.int32)
for i,l in enumerate(hh):
    l=[v for v in l if v>=0][-20:]
    if l: Hm[i,:len(l)]=l
uid=G.select(pl.col('user').rank('dense')-1)['user'].to_numpy().astype(np.int32)
np.savez('/tmp/dig/tr_ev.npz',cand=cand,lab=lab,H=Hm,uid=uid,ts=G['ts'].to_numpy(),tb=G['tb'].to_numpy(),user=G['user'].to_numpy())
print(len(G),'events; click rate',(lab<10).mean(),'empty hist',(Hm[:,0]<0).mean(), 'users',uid.max()+1)
# val sample tensors
D=pl.read_parquet('/tmp/dig/D.parquet',columns=['eid','user','item','clicked','cut'])
E=pl.read_parquet('/tmp/dig/events.parquet').select('eid','h_items','tb','ts')
De=D.select('eid').unique(maintain_order=True).join(E,on='eid',how='left')
hh=De.select(pl.col('h_items').list.eval(pl.element().replace_strict(imap['item'],imap['ix'],default=-1)))['h_items'].to_list()
Hv=np.full((len(hh),20),-1,np.int32)
for i,l in enumerate(hh):
    l=[v for v in l if v>=0][-20:]
    if l: Hv[i,:len(l)]=l
evix=D.join(De.with_row_index('k').select('eid','k'),on='eid',how='left',maintain_order='left')['k'].to_numpy()
cix=D.join(imap,on='item',how='left',maintain_order='left')['ix'].to_numpy()
np.savez('/tmp/dig/val_t.npz',H=Hv,evix=evix,cix=cix,user=D['user'].to_numpy(),evuser=D.select('eid','user').unique(maintain_order=True)['user'].to_numpy())
print('val rows',len(cix),'val events',len(Hv))
