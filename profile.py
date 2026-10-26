import duckdb, sys, os, glob, json, time
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
D=sys.argv[1] if len(sys.argv)>1 else '/workspace/huawei/data'
OUT=sys.argv[2] if len(sys.argv)>2 else '/workspace/huawei'
c=duckdb.connect(f'{OUT}/work.duckdb'); c.execute("SET memory_limit='3500MB'; SET threads=8; SET TimeZone='UTC'; SET preserve_insertion_order=false; SET temp_directory='/workspace/duck_tmp'")
q=lambda s: c.sql(s).fetchall()
df=lambda s: c.sql(s).df()
R={}
files=sorted(glob.glob(f'{D}/**/*',recursive=True)); files=[f for f in files if os.path.isfile(f)]
R['files']=[(os.path.relpath(f,D),os.path.getsize(f)) for f in files]
csvs=[f for f in files if f.lower().endswith(('.csv','.parquet','.tsv'))]
train=[f for f in csvs if 'train' in os.path.basename(f).lower()][0]
test=[f for f in csvs if any(k in os.path.basename(f).lower() for k in ('test','eval'))]
sub=[f for f in ['/workspace/huawei/example_submission.csv'] if os.path.exists(f)]
test=[f for f in test if f not in sub]
def src(f): return f"read_parquet('{f}')" if f.endswith('.parquet') else f"read_csv('{f}', header=true)"
def build(name,f):
    cols=[r[0] for r in q(f"DESCRIBE SELECT * FROM {src(f)}")]
    extra=[]
    if 'history_item_ids' in cols:
        extra.append("json(history_item_ids)::VARCHAR[] AS h_items")
    if 'history_timestamps' in cols:
        extra.append("json(history_timestamps)::BIGINT[] AS h_ts")
    sel=", ".join(["*"]+extra)
    t=time.time()
    c.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT {sel} FROM {src(f)}")
    print(name,'loaded',time.time()-t,flush=True)
build('train',train)
if test: build('test',test[0])
if sub: c.execute(f"CREATE OR REPLACE TABLE sub AS SELECT * FROM read_csv('{sub[0]}',header=true)")
def profile(t):
    P={}
    sch=q(f"DESCRIBE {t}"); P['schema']=[(a,b) for a,b,*_ in sch]
    n=q(f"SELECT count(*) FROM {t}")[0][0]; P['rows']=n
    rawcols=[a for a,b in P['schema'] if a not in ('h_items','h_ts')]
    P['ncols_raw']=len(rawcols)
    stats=[]
    for col,typ in P['schema']:
        if col in('h_items','h_ts'): continue
        nul,dist=q(f"SELECT count(*)-count({col}), approx_count_distinct({col}) FROM {t}")[0]
        exact=None
        if col not in ('history_item_ids','history_timestamps'):
            exact=q(f"SELECT count(DISTINCT {col}) FROM {t}")[0][0]
        emp=q(f"SELECT count(*) FILTER (WHERE trim({col}::VARCHAR) IN ('','[]','nan','NaN','null','None')) FROM {t}")[0][0]
        top=q(f"SELECT {col}::VARCHAR v, count(*) c FROM {t} GROUP BY 1 ORDER BY 2 DESC LIMIT 5") if col not in ('history_item_ids','history_timestamps') else []
        stats.append(dict(col=col,type=typ,null=nul,null_rate=nul/n,empty=emp,distinct=exact if exact is not None else dist,approx=exact is None,top=[(str(v)[:40],c_) for v,c_ in top]))
    P['cols']=stats
    num=[a for a,b in P['schema'] if b in('BIGINT','INTEGER','DOUBLE','FLOAT','SMALLINT','TINYINT','HUGEINT')]
    P['num']=df(f"SELECT * FROM (SUMMARIZE (SELECT {', '.join(num)} FROM {t}))")[['column_name','min','max','avg','std','q25','q50','q75']] if num else None
    if 'h_items' in [a for a,_ in P['schema']]:
        P['hist']=q(f"""SELECT min(len(h_items)),avg(len(h_items)),median(len(h_items)),max(len(h_items)),
          count(*) FILTER (WHERE len(h_items)=0), count(*) FILTER (WHERE len(h_items)<>len(h_ts)),
          count(*) FILTER (WHERE list_max(h_ts) > timestamp), count(*) FILTER (WHERE list_max(h_ts) = timestamp),
          count(*) FILTER (WHERE list_contains(h_items,item_id)),
          count(*) FILTER (WHERE h_ts <> list_sort(h_ts)),
          count(*) FILTER (WHERE len(list_distinct(h_items))<len(h_items))
          FROM {t}""")[0]
        P['histlen']=df(f"SELECT len(h_items) l, count(*) n FROM {t} GROUP BY 1 ORDER BY 1")
    if 'timestamp' in [a for a,_ in P['schema']]:
        P['trange']=q(f"SELECT to_timestamp(min(timestamp)),to_timestamp(max(timestamp)) FROM {t}")[0]
        P['ts_distinct']=q(f"SELECT count(DISTINCT timestamp) FROM {t}")[0][0]
    keycols=[a for a in rawcols]
    P['dup_full']=n-q(f"SELECT count(*) FROM (SELECT DISTINCT {', '.join(keycols)} FROM {t})")[0][0]
    nonid=[a for a in rawcols if a!='rowid']
    P['dup_norowid']=n-q(f"SELECT count(*) FROM (SELECT DISTINCT {', '.join(nonid)} FROM {t})")[0][0]
    if 'event_id' in rawcols and 'item_id' in rawcols:
        P['dup_event_item']=n-q(f"SELECT count(*) FROM (SELECT DISTINCT event_id,item_id FROM {t})")[0][0]
        P['ev']=q(f"""SELECT count(*), avg(k), min(k), max(k), median(k), max(nu), max(nts), max(nh) FROM (SELECT event_id, count(*) k,
            count(DISTINCT user_id) nu, count(DISTINCT timestamp) nts, count(DISTINCT history_item_ids) nh FROM {t} GROUP BY 1)""")[0]
        P['ev_size']=df(f"SELECT k, count(*) n FROM (SELECT event_id,count(*) k FROM {t} GROUP BY 1) GROUP BY 1 ORDER BY 2 DESC LIMIT 10")
    return P
TR=profile('train')
TE=profile('test') if test else None
# label analysis
lab='clicked'
L={}
L['ctr']=q(f"SELECT count(*), sum({lab}), avg({lab}) FROM train")[0]
L['labvals']=q(f"SELECT {lab}, count(*) FROM train GROUP BY 1 ORDER BY 1")
L['by_day']=df(f"SELECT to_timestamp(timestamp)::DATE d, count(*) n, avg({lab}) ctr FROM train GROUP BY 1 ORDER BY 1")
L['by_bucket']=df(f"SELECT time_bucket, count(*) n, avg({lab}) ctr FROM train GROUP BY 1 ORDER BY 1")
L['by_daytype']=df(f"SELECT day_type, count(*) n, avg({lab}) ctr FROM train GROUP BY 1 ORDER BY 1")
L['clicks_per_event']=df(f"SELECT s clicks, count(*) events FROM (SELECT event_id, sum({lab}) s FROM train GROUP BY 1) GROUP BY 1 ORDER BY 1 LIMIT 15")
L['by_histlen']=df(f"SELECT CASE WHEN len(h_items)=0 THEN '0' WHEN len(h_items)<5 THEN '1-4' WHEN len(h_items)<10 THEN '5-9' WHEN len(h_items)<20 THEN '10-19' ELSE '20+' END b, count(*) n, avg({lab}) ctr FROM train GROUP BY 1 ORDER BY min(len(h_items))")
L['in_hist']=df(f"SELECT list_contains(h_items,item_id) in_hist, count(*) n, avg({lab}) ctr FROM train GROUP BY 1")
L['users']=q(f"SELECT count(*), avg(n), max(n), avg(ctr), count(*) FILTER (WHERE ctr=0), count(*) FILTER (WHERE ctr=1) FROM (SELECT user_id, count(*) n, avg({lab}) ctr FROM train GROUP BY 1)")[0]
L['items']=q(f"SELECT count(*), avg(n), max(n), count(*) FILTER (WHERE n=1), count(*) FILTER (WHERE ctr=0) FROM (SELECT item_id, count(*) n, avg({lab}) ctr FROM train GROUP BY 1)")[0]
L['item_top']=df(f"SELECT item_id, count(*) n, avg({lab}) ctr FROM train GROUP BY 1 ORDER BY 2 DESC LIMIT 5")
# rowid pattern vs label (leakage-ish): check event-level order of positive within event via rowid order? check order in file
L['user_events']=q("SELECT avg(e), max(e) FROM (SELECT user_id, count(DISTINCT event_id) e FROM train GROUP BY 1)")[0]
# history items seen as train items
L['hist_item_cov']=q("SELECT count(*) FILTER (WHERE i.item_id IS NOT NULL)/count(*) FROM (SELECT DISTINCT unnest(h_items) h FROM train) h LEFT JOIN (SELECT DISTINCT item_id FROM train) i ON h.h=i.item_id")[0][0]
X={}
if TE:
    tc=set(a for a,_ in TE['schema']); trc=set(a for a,_ in TR['schema'])
    X['only_train']=sorted(trc-tc); X['only_test']=sorted(tc-trc)
    for k in ['user_id','item_id','event_id']:
        if k in tc:
            X[k]=q(f"SELECT avg((t.{k} IN (SELECT DISTINCT {k} FROM train))::INT), count(DISTINCT t.{k}) FROM test t")[0]
    X['time_overlap']=q("SELECT count(*) FILTER (WHERE timestamp BETWEEN (SELECT min(timestamp) FROM train) AND (SELECT max(timestamp) FROM train))/count(*) FROM test")[0][0]
    X['test_by_day']=df("SELECT to_timestamp(timestamp)::DATE d, count(*) n FROM test GROUP BY 1 ORDER BY 1")
    X['bucket']=df("SELECT time_bucket, count(*) n FROM test GROUP BY 1 ORDER BY 1")
S={}
if sub:
    S['file']=os.path.basename(sub[0]); S['schema']=q("DESCRIBE sub"); S['rows']=q("SELECT count(*) FROM sub")[0][0]
    S['head']=df("SELECT * FROM sub LIMIT 5")
    s0=S['schema'][0][0]
    S['distinct_first']=q(f"SELECT count(DISTINCT {s0}) FROM sub")[0][0]
    if TE and s0 in [a for a,_ in TE['schema']]:
        S['match_test']=q(f"SELECT count(*) FROM sub s JOIN test t USING({s0})")[0][0]
        S['same_order']=q(f"""WITH a AS (SELECT {s0}, row_number() OVER () r FROM sub), b AS (SELECT {s0}, row_number() OVER () r FROM test)
              SELECT avg((a.{s0}=b.{s0})::INT) FROM a JOIN b USING(r)""")[0][0]
    S['vals']=q(f"SELECT * FROM (SUMMARIZE sub)")
import pickle; pickle.dump(dict(TR=TR,TE=TE,L=L,X=X,S=S,R=R,train=train,test=test,sub=sub),open(f'{OUT}/profile.pkl','wb'))
# charts
fig,ax=plt.subplots(1,2,figsize=(12,4))
lv=L['labvals']; ax[0].bar([str(a) for a,_ in lv],[b for _,b in lv],color=['#888','#d33']); ax[0].set_title(f"Label balance (CTR={L['ctr'][2]:.4f})")
for a,b in lv: ax[0].text(str(a),b,f"{b:,}",ha='center',va='bottom')
bd=L['by_day']; ax2=ax[1]; ax2.bar(bd.d.astype(str),bd.n,color='#9bd'); ax2.set_ylabel('rows'); ax2.tick_params(axis='x',rotation=60)
ax3=ax2.twinx(); ax3.plot(bd.d.astype(str),bd.ctr,'r-o'); ax3.set_ylabel('CTR'); ax2.set_title('Rows and CTR per day (train)')
plt.tight_layout(); plt.savefig(f'{OUT}/charts/label_balance.png',dpi=110); plt.close()
fig,ax=plt.subplots(1,2,figsize=(12,4))
cs=TR['cols']; ax[0].barh([s['col'] for s in cs],[max(s['distinct'],1) for s in cs],color='#69c'); ax[0].set_xscale('log'); ax[0].set_title('Cardinality per column (log)')
ax[0].axvline(TR['rows'],color='r',ls='--',label='rows'); ax[0].legend()
hl=TR['histlen']; ax[1].bar(hl.l,hl.n,color='#7a7'); ax[1].set_title('History length distribution'); ax[1].set_xlabel('len(history_item_ids)')
plt.tight_layout(); plt.savefig(f'{OUT}/charts/cardinality_history.png',dpi=110); plt.close()
print('done')
