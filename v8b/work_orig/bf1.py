import polars as pl, sys, os, time
sys.path.insert(0,'/workspace/huawei/v8b/work_orig')
from feats import load, first_events, build
D='/workspace/huawei/v8b/work_orig'; os.makedirs(f'{D}/feat',exist_ok=True)
TE,TR,EE,ER,NI=load(D); DAY=86400; J20=1737331200
for name in sys.argv[1:]:
    t=time.time()
    if name=='eval': C=J20+7*DAY; T_ev=EE; T_rows=ER
    else: k=int(name[1:]); C=J20+k*DAY; T_ev=first_events(TE,C,DAY); T_rows=TR.join(T_ev.select('eid'),on='eid')
    build(C,T_ev,T_rows,TE,TR,NI).drop('h','h_ts').write_parquet(f'{D}/feat/{name}.parquet'); print(name, time.time()-t)
