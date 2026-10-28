import sys; from seqlib import *
from sklearn.metrics import roc_auc_score as A
kind=sys.argv[1]; ep=int(sys.argv[2]); loss=sys.argv[3] if len(sys.argv)>3 else 'gbce'; C=CUTS['C6']
t0=time.time(); m=train(kind,C,epochs=ep,loss=loss,log_=lambda s:print(s,flush=True)); print('train',time.time()-t0,flush=True)
tab=table_events('val26'); y=pl.read_parquet(f'{F3}/val26.parquet',columns=['clicked'])['clicked'].to_numpy(); t1=time.time(); o=featurize(m,tab,C,'val26'); print('feat',time.time()-t1)
for k,v in o.items(): vv=np.nan_to_num(v,nan=0.0); print(k,round(A(y,vv),5),round(A(y,-vv),5) if k in('rank','gap') else '')
