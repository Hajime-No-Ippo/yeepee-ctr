"""python seqfeat.py MODEL SEED CUTNAME TABLE... : fit MODEL (sas|gru; sampled-softmax) on click sequences ts<CUTNAME, write feat/{MODEL}_s{SEED}/{TABLE}.parquet (rowid + 10 cols prefixed {MODEL}_)"""
import sys,json; from seqlib import *
kind,seed,cn=sys.argv[1],int(sys.argv[2]),sys.argv[3]; tabs=sys.argv[4:]; C=CUTS[cn]; EP=int(os.environ.get('SEQ_EP',8 if kind=='sas' else 10))
t0=time.time(); m=train(kind,C,epochs=EP,loss='sce',seed=seed,log_=lambda s:print(s,flush=True)); tf=time.time()-t0
od=f'/workspace/huawei/v16seq/feat/{kind}_s{seed}'; os.makedirs(od,exist_ok=True)
for t in tabs:
    assert TABLE_CUT[t]==cn,(t,cn); t1=time.time(); tab=table_events(t); o=featurize(m,tab,C,t)
    pl.DataFrame({'rowid':tab['rowid'],**{f'{kind}_{k}':v for k,v in o.items()}}).write_parquet(f'{od}/{t}.parquet'); print(t,len(tab['rowid']),f'{time.time()-t1:.0f}s',flush=True)
r=dict(kind=kind,seed=seed,cut=cn,tables=tabs,epochs=EP,fit_s=round(tf),total_s=round(time.time()-t0)); print(json.dumps(r)); open('/workspace/huawei/v16seq/timings.jsonl','a').write(json.dumps(r)+'\n')
