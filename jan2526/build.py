import duckdb
c=duckdb.connect('j.duckdb')
c.execute("SET memory_limit='1500MB'; SET threads=4; SET temp_directory='/workspace/huawei/jan2526/tmp'; SET preserve_insertion_order=false")
c.execute("ATTACH '/workspace/huawei/work.duckdb' AS w (READ_ONLY)")
for src,lab in [('train','clicked'),('test','NULL::BIGINT')]:
    c.execute(f"""CREATE OR REPLACE TABLE r_{src} AS SELECT event_id, user_id, item_id, timestamp ts,
      {lab} AS clicked, len(h_items) hlen,
      len(list_filter(h_ts, lambda x: x < 1737331200)) nwarm,
      list_contains(h_items, item_id)::INT inhist,
      list_max(h_ts) last_h
      FROM w.{src}""")
    print(src, c.execute(f"select count(*) from r_{src}").fetchone())
