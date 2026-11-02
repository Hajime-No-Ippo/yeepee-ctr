exec(open('ev_members.py').read().split('def forms')[0])
for c in CUTS:
    d=D[c]; y=d['y']; b=roc_auc_score(y,d['base']); P=d['P']
    # event-level mean of M-logit per event (own slate) to build residual: P_ev vs M-implied
    import polars as pl
    rows=pl.read_parquet(f'rows_val{c}.parquet'); o=np.load(B+f'oof_c{c}.npz',allow_pickle=True)
    mp=dict(zip(rows['rowid'].to_numpy(),rows['eid'].to_numpy())); eid=np.array([mp[i] for i in o['rowid']])
    df=pl.DataFrame(dict(e=eid,M=d['M'])); 
    pm=1/(1+np.exp(-df['M'].to_numpy())); df=df.with_columns(pl.Series('pm',pm)); ev_any=df.group_by('e').agg((1-(1-pl.col('pm')).product()).alias('pa')); 
    pa=df.join(ev_any,on='e',how='left')['pa'].to_numpy()
    out=[]
    for w in (.002,.005,.01,.02):
        out.append(('A w%g'%w,roc_auc_score(y,(1-w)*d['base']+w*nr(P))-b))
    for a in (.05,.1,.2,.3):
        out.append(('C a%g'%a,roc_auc_score(y,nr(d['M']+a*lg(P)))-roc_auc_score(y,d['R']['M'])))
    for a in (.1,.25,.5):  # swap M's implied event-any prob for P_ev: add a*(logit P_ev - logit pa)
        out.append(('R a%g'%a,roc_auc_score(y,nr(d['M']+a*(lg(P)-lg(np.clip(pa,1e-4,1-1e-4)))))-roc_auc_score(y,d['R']['M'])))
    print(c,'corr(P_ev, M-implied any)=%.3f'%np.corrcoef(P,pa)[0,1],' '.join(f'{k}:{v*1e4:+.2f}' for k,v in out),flush=True)
