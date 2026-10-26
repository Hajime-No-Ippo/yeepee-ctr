import sys, time, polars as pl
sys.path.insert(0,'/workspace/huawei/v8b/prof')
from feats import load, first_events, build
D='/workspace/huawei/v8'; t=time.time(); TE,TR,EE,ER,NI=load(D); print('load',time.time()-t)
DAY=86400; J20=1737331200
for name in sys.argv[1:]:
    if name=='eval': C=J20+7*DAY; T_ev=EE; T_rows=ER
    else:
        k=int(name[1:]); C=J20+k*DAY; T_ev=first_events(TE,C,DAY); T_rows=TR.join(T_ev.select('eid'),on='eid')
    R=build(C,T_ev,T_rows,TE,TR,NI); print(name, R.shape)
