"""Adversarial validation (Eric-approved 30 Oct 2026 as DIAGNOSTIC ONLY): C25 training-fold rows (label 0) vs unlabelled eval rows (label 1), prefiltered set, LGB 200 rounds,
3-fold grouped by event. FLAG only (gain share>2% or single-feature AV AUC>0.6). Nothing is dropped from any tier. -> av_flagged.json (+ out/av_flags.json)."""
import os; os.environ.update(OMP_NUM_THREADS='4',POLARS_MAX_THREADS='4')
from lib20 import *
import numpy as np
wait_idle(); keep=json.load(open(f'{H}/out/redundancy.json'))['kept']; rs=np.random.RandomState(11)
parts=[]
for n in trn_tables(25):
    X,m,f=matrix(n,1.0,variant='notime',extra=EXTRA); ix=[f.index(c) for c in keep]; ne=len(X)//10; ev=np.sort(rs.choice(ne,15000,replace=False)); r=(ev[:,None]*10+np.arange(10)).ravel(); parts.append(X[r][:,ix]); del X,m
Xa=np.concatenate(parts); del parts
Xe,me,fe=matrix('eval',1.0,variant='notime',extra=EXTRA); Xe=Xe[:,[fe.index(c) for c in keep]]; del me
X=np.concatenate([Xa,Xe]); y=np.r_[np.zeros(len(Xa)),np.ones(len(Xe))]; ev_id=np.r_[np.arange(len(Xa))//10,10**6+np.arange(len(Xe))//10]; print(X.shape,flush=True)
fold=(ev_id*2654435761%2**32)%3; P=dict(objective='binary',learning_rate=0.1,num_leaves=63,min_data_in_leaf=200,feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10,verbose=-1,num_threads=4,max_bin=63,force_col_wise=True)
oof=np.zeros(len(y)); gain=np.zeros(len(keep)); aucs=[]
for k in range(3):
    wait_idle(); tr=fold!=k; ds=lgb.Dataset(X[tr],y[tr],feature_name=keep,params=dict(max_bin=63,force_col_wise=True)); m=lgb.train(P,ds,200)
    oof[~tr]=m.predict(X[~tr]); aucs.append(roc_auc_score(y[~tr],oof[~tr])); gain+=m.feature_importance('gain'); print('fold',k,aucs[-1],flush=True)
share=gain/gain.sum(); single={}
for j,c in enumerate(keep):
    v=np.nan_to_num(X[:,j],nan=-1e30); a=roc_auc_score(y,v); single[c]=max(a,1-a)
order=np.argsort(-share); flagged=[dict(feature=keep[j],av_gain_share=float(share[j]),single_feature_av_auc=float(single[keep[j]])) for j in order if share[j]>0.02 or single[keep[j]]>0.6]
out=dict(status='RUN (Eric approved as diagnostic; flag only, no feature dropped)',overall_av_auc=float(np.mean(aucs)),fold_aucs=aucs,n_train_rows=int(len(Xa)),n_eval_rows=int(len(Xe)),
  top20_by_gain=[dict(feature=keep[j],av_gain_share=float(share[j]),single_feature_av_auc=float(single[keep[j]])) for j in order[:20]],flagged=flagged,n_flagged=len(flagged),
  note='train sample = 15000 random events from each of f1..f4 (cut-25 training folds); eval=all 52,939 events. Eval rows used only for this diagnostic; no eval-derived feature enters any model.')
json.dump(out,open(f'{H}/av_flagged.json','w'),indent=1); json.dump(out,open(f'{H}/out/av_flags.json','w'),indent=1); print('AV AUC',out['overall_av_auc'],'flagged',len(flagged))
