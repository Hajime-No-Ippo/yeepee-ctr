import sys; sys.path.insert(0,'/workspace/huawei/parallel/src_snapshot/v2work')
from feats import load, first_events, build, log
D='/workspace/huawei/v2work'; DAY=86400; J20=1737331200
TE,TR,EE,ER,NI=load(D)
T_ev=first_events(TE,J20+5*DAY,2*DAY); T_rows=TR.join(T_ev.select('eid'),on='eid')
R=build(J20+4*DAY,T_ev,T_rows,TE,TR,NI)
R.drop('h','h_ts').write_parquet('/workspace/huawei/parallel/lag/feat/val25lag.parquet'); log('wrote',R.shape)
