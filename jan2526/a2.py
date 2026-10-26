import duckdb, pandas as pd, numpy as np
pd.set_option('display.width',250); pd.set_option('display.max_columns',40)
c=duckdb.connect('j.duckdb'); c.execute("SET memory_limit='1500MB'; SET threads=4")
T0=1737331200
# user activity profile
c.execute("""CREATE OR REPLACE TABLE uprof AS SELECT user_id,
  min(d) FILTER (src='train') first_d, count(*) FILTER (src='train') n_tr_ev,
  count(*) FILTER (src='train' AND d<5) n_wd, count(*) FILTER (src='train' AND d>=5) n_we,
  min(ts) FILTER (src='train') first_ts
  FROM ev GROUP BY user_id""")
day = c.execute(f"""
SELECT e.d, count(*) events, count(*)*10 rows_, count(DISTINCT e.user_id) users,
  count(*)/count(DISTINCT e.user_id) ev_per_user, avg(clk) ev_click_rate, avg(clk)/10 row_ctr,
  avg(hlen) hlen_mean, avg((hlen=0)::INT) empty_hist, sum(nwarm)/sum(hlen) warm_share,
  avg(ninh)/10 inhist_rate, avg((ninh>0)::INT) ev_with_inhist
FROM ev e GROUP BY d ORDER BY d""").df()
# users per day: weekend-only vs weekday too, new-to-window
u = c.execute("""
WITH du AS (SELECT DISTINCT d,user_id FROM ev)
SELECT du.d, count(*) users,
 avg((p.n_wd=0)::INT) weekend_only_share, avg((p.n_we=0)::INT) weekday_only_share,
 avg((p.first_d=du.d)::INT) first_day_in_window_share, avg((p.n_tr_ev=0)::INT) not_in_train_share
FROM du JOIN uprof p USING(user_id) GROUP BY du.d ORDER BY du.d""").df()
day = day.merge(u.drop(columns='users'), on='d')
# row CTR by inhist
ih = c.execute(f"""SELECT (ts-{T0})//86400 d, avg(clicked) FILTER (inhist=1) ctr_inhist, avg(clicked) FILTER (inhist=0) ctr_notinhist FROM r_train GROUP BY 1 ORDER BY 1""").df()
day = day.merge(ih, on='d', how='left')
day.insert(0,'day', ['Mon20','Tue21','Wed22','Thu23','Fri24','Sat25','Sun26','EVAL'])
# weekday pooled and weekend pooled rows
print(day.T.to_string())
day.to_csv('per_day.csv',index=False)
# hour mix
hr = c.execute("SELECT d, (ts%86400)//3600 h, count(*) n FROM ev GROUP BY all").df()
hr = hr.pivot(index='h',columns='d',values='n').fillna(0); hr = hr/hr.sum()
hr.to_csv('hour_mix.csv')
tb = hr.groupby(hr.index//6).sum(); print((tb*100).round(1).to_string())
# eval date mix
print(c.execute("SELECT (ts-1737936000)//86400 dd, count(*) , avg(((ts-1737936000)//86400 % 7 in (5,6))::INT) FROM ev WHERE src='eval' GROUP BY 1 ORDER BY 1").df().to_string())
# hourly event click rate train
print(c.execute("SELECT d>=5 we, (ts%86400)//3600 h, avg(clk) cr FROM ev WHERE src='train' GROUP BY all ORDER BY all").df().pivot(index='h',columns='we',values='cr').round(3).T.to_string())
