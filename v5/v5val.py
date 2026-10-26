"""v5 validation runner (same model/feature path as combo.py val, shipped config: notime, XF=1 XF2=1, f-folds).
Usage: python v5val.py TAG STEP CUT [CUT...]
env: SEEDS=42,43,44 ROUNDS=400 XDIRS=feat_x,feat_x2 (dirs joined on rowid) DROPF=regex (drop feats) ADDF=dir1,dir2 (extra dirs)
Appends one line per cut to out/results.jsonl; skips (tag,cut) already present."""
import sys, os, json, time, re, gc, numpy as np, polars as pl, lightgbm as lgb
sys.path.insert(0,'/workspace/huawei/v5'); sys.path.insert(1,'/workspace/huawei/v3')
from feats import DROP
from features import feature_cols, GROUPS
from sklearn.metrics import roc_auc_score
D='/workspace/huawei/v5'; P2=f'{D}/feat'; P3=f'{D}/feat_v3'; RES=f'{D}/out/results.jsonl'
V2_OF={'f1':'h1','f2':'h2','f3':'h3','f4':'h4','f5':'h5','f6':'val26','val23':'val23','val25':'val25','val26':'val26','eval':'eval'}
TIME=['tm_stale','tm_hour','tm_tb','tm_dt','v2_tb','v2_dt','v2_stale','v2_hour']
XDIRS=[x for x in os.environ.get('XDIRS','feat_x,feat_x2').split(',') if x]+[x for x in os.environ.get('ADDF','').split(',') if x]
DROPF=os.environ.get('DROPF'); SEEDS=[int(s) for s in os.environ.get('SEEDS','42,43,44').split(',')]; ROUNDS=int(os.environ.get('ROUNDS','400'))
PARAMS=dict(objective='binary',learning_rate=0.05,num_leaves=63,min_data_in_leaf=200,feature_fraction=0.7,bagging_fraction=0.8,
            bagging_freq=1,lambda_l2=10.0,verbose=-1,num_threads=3)
def load(n):
    a=pl.read_parquet(f'{P3}/{n}.parquet'); b=pl.read_parquet(f'{P2}/{V2_OF[n]}.parquet')
    f2=[c for c in b.columns if c not in DROP]
    a=a.join(b.select(['rowid']+f2).rename({c:'v2_'+c for c in f2}),on='rowid',how='left')
    for x in XDIRS: a=a.join(pl.read_parquet(f'{D}/{x}/{n}.parquet'),on='rowid',how='left')
    return a
def featlist(df):
    f=feature_cols(df,GROUPS)+[c for c in df.columns if c.startswith(('v2_','x_','r_','s_','sl_'))]
    f=[c for c in f if c not in TIME]
    if DROPF: f=[c for c in f if not re.search(DROPF,c)]
    return f
def evaucs(eid,y,s):
    df=pl.DataFrame({'e':eid,'y':y,'s':s})
    E=df.group_by('e').agg(pl.col('y').max(),pl.col('s').sum())
    W=df.group_by('e').agg((pl.col('y').max()>pl.col('y').min()).alias('mx'),pl.col('y'),pl.col('s')).filter('mx')
    wa=np.mean([roc_auc_score(np.array(r[0]),np.array(r[1])) for r in W.select('y','s').iter_rows()])
    return round(roc_auc_score(E['y'],E['s']),5), round(float(wa),5)
tag,step=sys.argv[1],sys.argv[2]; cuts=[int(c) for c in sys.argv[3:]]
done=set()
if os.path.exists(RES):
    for l in open(RES):
        try: r=json.loads(l); done.add((r.get('tag'),r.get('cut')))
        except Exception: pass
for cut in cuts:
    if (tag,cut) in done: print('skip',tag,cut,flush=True); continue
    t0=time.time(); feats=None; Xs=[]; ys=[]
    for k in range(1,cut-20):
        t=load(f'f{k}')
        if feats is None: feats=featlist(t)
        Xs.append(t.select(feats).to_numpy().astype(np.float32)); ys.append(t['clicked'].to_numpy()); del t; gc.collect()
    X=np.concatenate(Xs); del Xs; gc.collect()
    ds=lgb.Dataset(X,np.concatenate(ys),feature_name=feats,free_raw_data=True).construct(); del X; gc.collect()
    va=load(f'val{cut}'); Xv=va.select(feats).to_numpy().astype(np.float32); y=va['clicked'].to_numpy()
    eid=va['eid'].to_numpy(); dow=((va['ts'].to_numpy()//86400)+3)%7
    cold=va['us_cold'].to_numpy()==1; noh=va['us_nohist'].to_numpy()==1; st=va['tm_stale'].to_numpy()
    segs={'warm':~cold,'history_only':cold&~noh,'empty_history':noh,'weekday':dow<5,'weekend':dow>=5,'stale_lt1d':st<86400,'stale_ge1d':st>=86400}
    per=[]; P={300:0,ROUNDS:0}; G=np.zeros(len(feats))
    for s in SEEDS:
        m=lgb.train(dict(PARAMS,seed=s),ds,num_boost_round=ROUNDS)
        p3=m.predict(Xv,num_iteration=300); p4=m.predict(Xv,num_iteration=ROUNDS)
        P[300]=P[300]+p3/len(SEEDS); P[ROUNDS]=P[ROUNDS]+p4/len(SEEDS)
        g=m.feature_importance('gain'); G+=g/g.sum()/len(SEEDS)
        seg={k:(round(roc_auc_score(y[v],p4[v]),5) if v.sum()>0 and y[v].min()<y[v].max() else None) for k,v in segs.items()}
        ea,wa=evaucs(eid,y,p4)
        per.append(dict(seed=s,auc300=round(roc_auc_score(y,p3),5),auc400=round(roc_auc_score(y,p4),5),seg=seg,event_auc=ea,within_auc=wa))
        np.save(f'{D}/out/pred_{tag}_c{cut}_s{s}.npy',p4.astype(np.float32)); print(tag,cut,per[-1],flush=True)
    mean=lambda key: round(float(np.mean([r[key] for r in per])),5)
    mseg={k:(round(float(np.mean([r['seg'][k] for r in per])),5) if per[0]['seg'][k] is not None else None) for k in segs}
    res=dict(tag=tag,step=step,cut=cut,seeds=SEEDS,rounds=ROUNDS,n_feats=len(feats),xdirs=XDIRS,dropf=DROPF,
             auc300_mean=mean('auc300'),auc400_mean=mean('auc400'),event_auc_mean=mean('event_auc'),within_auc_mean=mean('within_auc'),seg_mean=mseg,
             seg_n={k:int(v.sum()) for k,v in segs.items()},auc400_seedavg_pred=round(roc_auc_score(y,P[ROUNDS]),5),per_seed=per,
             top15=[(feats[i],round(float(G[i]),4)) for i in np.argsort(-G)[:15]],runtime_s=round(time.time()-t0),
             ts=time.strftime('%Y-%m-%d %H:%M:%S %Z'))
    open(RES,'a').write(json.dumps(res)+'\n'); json.dump(feats,open(f'{D}/out/feats_{tag}.json','w'))
    print('RESULT',json.dumps({k:res[k] for k in ('tag','cut','n_feats','auc400_mean','event_auc_mean','runtime_s')}),flush=True)
    del ds,va,Xv; gc.collect()
