import polars as pl, time, sys
D='/workspace/huawei/lite'
def hexid(e, n=15):  # opaque id -> int64 via first 15 hex chars after prefix
    return e.str.slice(4, n).str.to_integer(base=16)
def prep(name, has_label):
    t=time.time()
    lf=pl.scan_csv(f'{D}/{name}.csv', schema_overrides={'history_item_ids':pl.Utf8,'history_timestamps':pl.Utf8})
    cols=[pl.col('rowid'), hexid(pl.col('event_id')).alias('eid'), hexid(pl.col('item_id')).alias('item')]
    if has_label: cols.append(pl.col('clicked').cast(pl.Int8))
    lf.select(cols).sink_parquet(f'{D}/{name}_rows.parquet')
    print(name,'rows done',time.time()-t,flush=True)
    ev=(lf.select('event_id','user_id','timestamp','time_bucket','day_type','history_item_ids','history_timestamps')
          .unique(subset='event_id',keep='any')
          .select(hexid(pl.col('event_id')).alias('eid'), hexid(pl.col('user_id')).alias('user'),
                  pl.col('timestamp').cast(pl.Int64).alias('ts'), pl.col('time_bucket').cast(pl.Int8).alias('tb'),
                  pl.col('day_type').cast(pl.Int8).alias('dt'),
                  pl.col('history_item_ids').str.json_decode(pl.List(pl.Utf8)).list.eval(pl.element().str.slice(4,15).str.to_integer(base=16)).alias('h_items'),
                  pl.col('history_timestamps').str.json_decode(pl.List(pl.Int64)).alias('h_ts')))
    ev.collect(engine='streaming').sort('ts').write_parquet(f'{D}/{name}_events.parquet')
    print(name,'events done',time.time()-t,flush=True)
prep('evaluation',False)
prep('train',True)
