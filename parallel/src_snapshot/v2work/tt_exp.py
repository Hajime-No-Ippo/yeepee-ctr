import polars as pl, numpy as np, sys, time
sys.path.insert(0,'/workspace/huawei/v2work')
from feats import load, first_events
import tt
from sklearn.metrics import roc_auc_score
TE,TR,EE,ER,NI=load('/workspace/huawei/v2work')
DAY=86400; C=1737331200+4*DAY
T_ev=first_events(TE,C,3*DAY); T_rows=TR.join(T_ev.select('eid'),on='eid')
P_ev=TE.filter(pl.col('ts')<C); P_rows=TR.join(P_ev.select('eid'),on='eid')
ep=int(sys.argv[1]) if len(sys.argv)>1 else 3; d=int(sys.argv[2]) if len(sys.argv)>2 else 64
m=tt.train(P_ev,P_rows,NI,epochs=ep,d=d,wd=float(sys.argv[3]) if len(sys.argv)>3 else 1e-6)
out=tt.score(m,T_ev,T_rows).join(T_rows.select('eid',pl.col('ii').cast(pl.Int32),'clicked'),on=['eid','ii'])
out=out.with_columns((pl.col('tt_score')-pl.col('tt_score').mean().over('eid')).alias('dm'))
y=out['clicked'].to_numpy()
print('TT',ep,d,'auc',roc_auc_score(y,out['tt_score']),'dm',roc_auc_score(y,out['dm']))
