import duckdb, pandas as pd, numpy as np
from scipy.stats import spearmanr, pearsonr
c=duckdb.connect('j.duckdb'); c.execute("SET memory_limit='1500MB'; SET threads=4")
T0=1737331200
imp = c.execute(f"""SELECT item_id, (ts-{T0})//86400 d, count(*) n, sum(clicked) k, (hash(event_id)%2) half FROM r_train GROUP BY all""").df()
ev = c.execute("SELECT item_id, count(*) n FROM r_test GROUP BY 1").df().set_index('item_id').n
items = sorted(set(imp.item_id)|set(ev.index))
I = pd.pivot_table(imp, index='item_id', columns='d', values='n', aggfunc='sum').reindex(items).fillna(0)
K = pd.pivot_table(imp, index='item_id', columns='d', values='k', aggfunc='sum').reindex(items).fillna(0)
H = pd.pivot_table(imp, index='item_id', columns=['d','half'], values='n', aggfunc='sum').reindex(items).fillna(0)
E = ev.reindex(items).fillna(0)
print('items', len(items), 'eval items', (E>0).sum(), 'per-day items', (I>0).sum().tolist())
def js(p,q):
    p=p/p.sum(); q=q/q.sum(); m=(p+q)/2
    f=lambda a: np.sum(a[a>0]*np.log2(a[a>0]/m[a>0]))
    return 0.5*f(p)+0.5*f(q)
def topshare(x,k=100): x=np.sort(x)[::-1]; return x[:k].sum()/x.sum()
out=[]
wd=[0,1,2,3,4]
for d in range(7):
    ref = I[[w for w in wd if w!=d]].sum(1).values
    refk = K[[w for w in wd if w!=d]].sum(1).values
    x=I[d].values; k=K[d].values
    # noise floor: two halves of the same day
    h0=H[(d,0)].values; h1=H[(d,1)].values
    out.append(dict(d=d, top100_imp=topshare(x), top1pct_imp=topshare(x,500), top100_clk=topshare(k),
      js_imp_vs_wd=js(x,ref), js_imp_vs_eval=js(x,E.values), js_imp_halves_floor=js(h0,h1),
      js_clk_vs_wd=js(k,refk),
      sp_imp_vs_wd=spearmanr(x,ref)[0], pe_imp_vs_wd=pearsonr(x,ref)[0], sp_imp_vs_eval=spearmanr(x,E.values)[0], pe_imp_vs_eval=pearsonr(x,E.values)[0],
      pe_clk_vs_wd=pearsonr(k,refk)[0],
      pe_itemctr_vs_wd=pearsonr(np.where(x>=200,k/np.maximum(x,1),np.nan)[(x>=200)&(ref>=800)], (refk/np.maximum(ref,1))[(x>=200)&(ref>=800)])[0]))
wdall=I[wd].sum(1).values
# eval self: compare eval with weekday pooled, weekend pooled
o=pd.DataFrame(out)
o.loc[len(o)]=dict(d=7, top100_imp=topshare(E.values), top1pct_imp=topshare(E.values,500), js_imp_vs_wd=js(E.values,wdall), sp_imp_vs_wd=spearmanr(E.values,wdall)[0], pe_imp_vs_wd=pearsonr(E.values,wdall)[0])
# subsample floor: a weekday subsampled to eval size vs eval? -> JS of random 529k-row subsample of day 24 vs weekday pool
rng=np.random.default_rng(0); x=I[4].values.astype(int)
sub=rng.multivariate_hypergeometric(x, int(E.sum()))
print('JS floor: Fri24 subsampled to eval size vs other weekdays', js(sub, I[[0,1,2,3]].sum(1).values), ' vs eval', js(sub,E.values))
print('JS Sat+Sun pooled vs Mon-Fri pooled', js(I[[5,6]].sum(1).values, wdall), 'Thu+Fri vs Mon-Wed', js(I[[3,4]].sum(1).values, I[[0,1,2]].sum(1).values))
print('JS eval vs Sat+Sun', js(E.values, I[[5,6]].sum(1).values), 'eval vs Fri24', js(E.values,I[4].values), 'eval vs Sun26', js(E.values,I[6].values),'eval vs Mon20', js(E.values,I[0].values))
print('pearson eval vs Sun26', pearsonr(E.values,I[6].values)[0], 'eval vs Mon20', pearsonr(E.values,I[0].values)[0])
o.insert(0,'day',['Mon20','Tue21','Wed22','Thu23','Fri24','Sat25','Sun26','EVAL'])
pd.set_option('display.width',250)
print(o.T.to_string())
o.to_csv('items_per_day.csv',index=False)
I.assign(eval=E.values).to_parquet('item_day_imp.parquet')
