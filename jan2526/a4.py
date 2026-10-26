import duckdb, pandas as pd, numpy as np
c=duckdb.connect('j.duckdb'); c.execute("SET memory_limit='1500MB'; SET threads=4")
T0=1737331200
rows=[]; dists={}
for C in [21,22,23,24,25,26,27]:
    cut=T0+(C-20)*86400
    cond = "src='eval'" if C==27 else f"src='train' AND ts>={cut}"
    q=f"""WITH f AS (SELECT * FROM (SELECT *, row_number() OVER (PARTITION BY user_id ORDER BY ts, event_id) rn FROM ev WHERE {cond}) WHERE rn=1),
    prev AS (SELECT user_id, max(ts) last_ev, count(*) n_prev FROM ev WHERE src='train' AND ts<{cut} GROUP BY 1)
    SELECT f.*, ({'f.ts' } - {cut})/3600.0 stale_h, coalesce(n_prev,0) n_prev, (f.ts-last_ev)/3600.0 gap_h, (f.ts-f.last_h)/3600.0 since_click_h,
      ((f.ts-{T0})//86400) % 7 IN (5,6) is_we, (f.ts%86400)//3600 hr
    FROM f LEFT JOIN prev USING(user_id)"""
    df=c.execute(q).df()
    dists[C]=df[['stale_h','gap_h','since_click_h','hr','hlen','is_we','clk','ninh']]
    rows.append(dict(set=('EVAL' if C==27 else f'val C={C}'), events=len(df), rows=len(df)*10,
      stale_med_h=df.stale_h.median(), stale_mean_h=df.stale_h.mean(), stale_p90_h=df.stale_h.quantile(.9), stale_gt24h=(df.stale_h>=24).mean(), stale_gt48h=(df.stale_h>=48).mean(),
      weekend_share=df.is_we.mean(), tb0=(df.hr<6).mean(), tb1=((df.hr>=6)&(df.hr<12)).mean(), tb2=((df.hr>=12)&(df.hr<18)).mean(), tb3=(df.hr>=18).mean(),
      hlen_mean=df.hlen.mean(), hlen20_share=(df.hlen==20).mean(), empty_hist=(df.hlen==0).mean(), warm_share=df.nwarm.sum()/df.hlen.sum(),
      inhist_rate=df.ninh.mean()/10, new_user_share=(df.n_prev==0).mean(), gap_med_h=df.gap_h.median(), since_click_med_h=df.since_click_h.median(),
      ev_click_rate=df.clk.mean() if C<27 else np.nan))
v=pd.DataFrame(rows).set_index('set')
pd.set_option('display.width',250)
print(v.T.round(4).to_string())
v.to_csv('valsets.csv')
pd.to_pickle(dists,'val_dists.pkl')
