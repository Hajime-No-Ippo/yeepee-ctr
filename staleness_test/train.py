# Staleness test: train/evaluate baseline, +A, +B, +A+B (+variants) on folds C=24, C=25.
import numpy as np, polars as pl, lightgbm as lgb, json, sys, time
from sklearn.metrics import roc_auc_score
O='/workspace/huawei/staleness_test/feat'; R='/workspace/huawei/staleness_test'
SEEDS=[0,1,2]; ROUNDS=int(sys.argv[1]) if len(sys.argv)>1 else 400
EVAL_MIX=np.array([0.67,0.165,0.165])
def bucket(s): return np.digitize(s,[24,48])  # 0:<24h 1:24-48h 2:>=48h
def rew(stale):
    b=bucket(stale); share=np.bincount(b,minlength=3)/len(b); tgt=EVAL_MIX.copy()
    if share[2]==0: tgt=np.array([0.67,0.33,0.0])   # C=25 has no >=48h rows: collapse
    return (tgt/np.where(share>0,share,1))[b], tgt
def load(nms): return pl.concat([pl.read_parquet(f'{O}/{n}.parquet') for n in nms])
META=['y','e','stale_h']
t0=time.time(); res=[]; preds={}
for C in (25,24):
    Ps=[C-3,C-2,C-1]
    tr={'plain':load([f'plain_P{p}' for p in Ps]), 'B':load([f'B_P{p}_C{C}' for p in Ps])}
    va=load([f'val_C{C}']); yv=va['y'].to_numpy(); sv=va['stale_h'].to_numpy(); wv,tgt=rew(sv); bv=bucket(sv)
    allf=[c for c in va.columns if c not in META]
    base=[c for c in allf if not c.startswith('A_')]
    Aset=base+[c for c in allf if c.startswith('A_')]+['stale_h']
    Atime=base+['A_since_click_h','A_click_to_cut_h','A_since_ev_h','A_hist24','A_hist_span_h','A_inh_x_stale','A_inh_age_rel','stale_h']
    variants={'base':('plain',base,None),'+A':('plain',Aset,None),'+Atime':('plain',Atime,None),
              '+B':('B',base,None),'+A+B':('B',Aset,None),'+A+B_w':('B',Aset,'w'),'+B_w':('B',base,'w')}
    for vn,(ts,feats,wt) in variants.items():
        d=tr[ts]; X=d.select(feats).to_numpy().astype(np.float32); y=d['y'].to_numpy()
        w=None
        if wt=='w': w,_=rew(d['stale_h'].to_numpy()); 
        Xv=va.select(feats).to_numpy().astype(np.float32); P=[]
        for s in SEEDS:
            p=dict(objective='binary',learning_rate=0.05,num_leaves=63,min_data_in_leaf=200,feature_fraction=0.8,
                   bagging_fraction=0.8,bagging_freq=1,lambda_l2=10,num_threads=4,seed=s,verbose=-1,max_bin=127)
            m=lgb.train(p,lgb.Dataset(X,y,weight=w),ROUNDS); pv=m.predict(Xv); P.append(pv)
            r=dict(C=C,variant=vn,seed=s,ntrain=len(y),train_stale24=float((d['stale_h'].to_numpy()>=24).mean()),
                   auc=roc_auc_score(yv,pv),auc_rw=roc_auc_score(yv,pv,sample_weight=wv))
            for b in range(3):
                mb=bv==b
                if mb.sum()>0: r[f'auc_b{b}']=roc_auc_score(yv[mb],pv[mb])
            res.append(r); print(f'[{time.time()-t0:5.0f}s]',json.dumps({k:(round(v,5) if isinstance(v,float) else v) for k,v in r.items()}),flush=True)
        preds[(C,vn)]=np.array(P)
    np.savez_compressed(f'{R}/preds_C{C}.npz',y=yv,stale=sv,e=va['e'].to_numpy(),**{vn:preds[(C,vn)] for vn in variants})
    if C==25: 
        imp=dict(zip(feats,m.feature_importance('gain').round().tolist())); json.dump(imp,open(f'{R}/imp_last.json','w'))
    del tr,va
pl.DataFrame(res).write_csv(f'{R}/results_raw.csv'); print('done')
