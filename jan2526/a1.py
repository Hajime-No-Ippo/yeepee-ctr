import duckdb, pandas as pd, numpy as np, json
pd.set_option('display.width',250); pd.set_option('display.max_columns',40)
c=duckdb.connect('j.duckdb')
c.execute("SET memory_limit='1500MB'; SET threads=4; SET temp_directory='/workspace/huawei/jan2526/tmp'")
T0=1737331200
c.execute(f"""CREATE OR REPLACE VIEW rr AS
 SELECT *, 'train' src, (ts-{T0})//86400 AS d FROM r_train
 UNION ALL SELECT *, 'eval' src, 7 AS d FROM r_test""")
# event table
c.execute(f"""CREATE OR REPLACE TABLE ev AS SELECT event_id, any_value(user_id) user_id, any_value(ts) ts, any_value(d) d, any_value(src) src,
  max(clicked) clk, count(*) n, any_value(hlen) hlen, any_value(nwarm) nwarm, sum(inhist) ninh, any_value(last_h) last_h
  FROM rr GROUP BY event_id""")
print(c.execute("select src,d,count(*),min(n),max(n) from ev group by all order by d").df())
