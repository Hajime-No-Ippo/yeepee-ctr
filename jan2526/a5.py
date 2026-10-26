import duckdb, pandas as pd, numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import svds
from sklearn.metrics import roc_auc_score
c=duckdb.connect('j.duckdb'); c.execute("SET memory_limit='1500MB'; SET threads=4")
c.execute("ATTACH '/workspace/huawei/work.duckdb' AS w (READ_ONLY)")
T0=1737331200
items=c.execute("SELECT DISTINCT item_id FROM r_train ORDER BY 1").df().item_id
imap={x:i for i,x in enumerate(items)}; NI=len(items)
users=c.execute("SELECT DISTINCT user_id FROM ev ORDER BY 1").df().user_id
umap={x:i for i,x in enumerate(users)}
# all known clicks: warm-up entries from each user's earliest train row + train clicked rows
warm=c.execute(f"""WITH f AS (SELECT user_id, arg_min(h_items, timestamp) hi, arg_min(h_ts,timestamp) ht FROM w.train GROUP BY user_id)
 SELECT user_id, unnest(hi) item_id, unnest(ht) ts FROM f""").df()
warm=warm[warm.ts<T0]
clk=c.execute("SELECT user_id,item_id,ts FROM r_train WHERE clicked=1").df()
allc=pd.concat([warm,clk]); allc['u']=allc.user_id.map(umap); allc['i']=allc.item_id.map(imap)
print('clicks', len(warm), len(clk))
def fit(cut):
    a=allc[allc.ts<cut]
    M=sp.csr_matrix((np.ones(len(a)),(a.u,a.i)),shape=(len(users),NI)); M.data[:]=1
    # popularity: clicks in the 24h before cut (equal-length window)
    b=a[a.ts>=cut-86400]; pop=np.bincount(b.i,minlength=NI).astype(float)
    # normalized co-occurrence SVD
    ic=np.asarray(M.sum(0)).ravel()+1; uc=np.asarray(M.sum(1)).ravel()+1
    Mn=sp.diags(1/np.sqrt(uc))@M@sp.diags(1/np.sqrt(ic))
    U,s,Vt=svds(Mn.astype(np.float32),k=16)
    V=(Vt.T*s); V/=np.linalg.norm(V,axis=1,keepdims=True)+1e-9
    return pop,V
def score(df,pop,V):
    ii=df.item_id.map(imap).values
    sim=np.zeros(len(df))
    for r,(hi,it) in enumerate(zip(df.h_items.values, ii)):
        if len(hi)==0: continue
        hv=V[[imap[x] for x in hi]].mean(0); n=np.linalg.norm(hv)
        if n>0: sim[r]=V[it]@hv/n
    return dict(inhist=df.inhist.values.astype(float), pop=np.log1p(pop[ii]), sim=sim)
def aucs(df,S):
    out={}
    for k,v in S.items():
        out['auc_'+k]=roc_auc_score(df.clicked,v)
        # within-event AUC via groupby on events with a click
        g=pd.DataFrame(dict(e=df.event_id.values,y=df.clicked.values,s=v))
        g['r']=g.groupby('e').s.rank(); g=g[g.groupby('e').y.transform('sum')==1]
        pos=g[g.y==1].r.values; out['wauc_'+k]=np.mean((pos-1)/9)
    return out
res=[]
for d in range(1,7):
    cut=T0+d*86400; pop,V=fit(cut)
    df=c.execute(f"""SELECT r.event_id,r.item_id,r.clicked,r.inhist,t.h_items FROM r_train r JOIN w.train t ON t.rowid IS NOT NULL AND false""").df() if False else None
    df=c.execute(f"""WITH e AS (SELECT event_id FROM ev WHERE src='train' AND d={d} AND hash(event_id)%4=0)
      SELECT t.event_id,t.item_id,t.clicked,list_contains(t.h_items,t.item_id)::INT inhist,t.h_items FROM w.train t SEMI JOIN e USING(event_id)""").df()
    r=aucs(df,score(df,pop,V)); r.update(set=f'day {20+d}', n_ev=df.event_id.nunique(), weekend=d>=5); res.append(r); print(r,flush=True)
for C in [24,25,26]:
    cut=T0+(C-20)*86400; pop,V=fit(cut)
    df=c.execute(f"""WITH f AS (SELECT event_id, ts FROM (SELECT *, row_number() OVER (PARTITION BY user_id ORDER BY ts,event_id) rn FROM ev WHERE src='train' AND ts>={cut}) WHERE rn=1)
      SELECT t.event_id,t.item_id,t.clicked,list_contains(t.h_items,t.item_id)::INT inhist,t.h_items,(t.timestamp-{cut})/3600.0 st, ((t.timestamp-{T0})//86400)>=5 we FROM w.train t JOIN f USING(event_id)""").df()
    S=score(df,pop,V)
    r=aucs(df,S); r.update(set=f'val C={C}', n_ev=df.event_id.nunique()); res.append(r); print(r,flush=True)
    for lab,m in [('stale<24h',df.st<24),('stale>=24h',df.st>=24),('weekday ev',~df.we),('weekend ev',df.we)]:
        if m.sum()<1000: continue
        sub=df[m.values].reset_index(drop=True); r=aucs(sub,{k:v[m.values] for k,v in S.items()}); r.update(set=f'val C={C} {lab}', n_ev=sub.event_id.nunique()); res.append(r); print(r,flush=True)
pd.DataFrame(res).to_csv('signal_auc.csv',index=False)
