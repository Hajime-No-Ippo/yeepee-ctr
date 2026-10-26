"""Combined (v3 features + v2 features) LightGBM. Modes:
  python combo.py val 25|26 [variant]   -> fixed-round curve on val25/val26, segment + weekday/weekend AUCs
  python combo.py final N_ITER SEEDS    -> fit on all 6 cutoff folds (Jan21..Jan26), predict eval (cutoff Jan27)
variants: base | notime (drop all time/gap feats) | wkw (weekend train events weight 0.3) | notime_wkw"""
import sys, os, json, numpy as np, polars as pl, lightgbm as lgb
sys.path.insert(0,'/workspace/huawei/v2work'); sys.path.insert(0,'/workspace/huawei/v3')
from feats import DROP
from features import feature_cols, GROUPS
from sklearn.metrics import roc_auc_score
P2='/workspace/huawei/v2work/feat'; P3='/workspace/huawei/v2work/feat_v3'
V2_OF={'f1':'h1','f2':'h2','f3':'h3','f4':'h4','f5':'h5','f6':'val26','val25':'val25','val26':'val26','eval':'eval'}
V2_OF.update({f'{p}{k}':f'{p}{k}' for p in 'eE' for k in range(1,7)})
EFT=os.environ.get('FOLDS')=='eft'   # eval-like multi-day target windows (e1..e4 -> Jan25, E1..E6 -> Jan27)
OUTP=os.environ.get('OUTP','eval_pred')
TIME=['tm_stale','tm_hour','tm_tb','tm_dt','v2_tb','v2_dt','v2_stale','v2_hour']
def load(names):
    out=[]
    for n in names:
        a=pl.read_parquet(f'{P3}/{n}.parquet'); b=pl.read_parquet(f'{P2}/{V2_OF[n]}.parquet')
        f2=[c for c in b.columns if c not in DROP]
        b=b.select(['rowid']+f2).rename({c:'v2_'+c for c in f2})
        a=a.join(b,on='rowid',how='left')
        if os.environ.get('XF'):
            a=a.join(pl.read_parquet(f'/workspace/huawei/v2work/feat_x/{n}.parquet'),on='rowid',how='left')
        if os.environ.get('XF2'):
            a=a.join(pl.read_parquet(f'/workspace/huawei/v2work/feat_x2/{n}.parquet'),on='rowid',how='left')
        out.append(a)
    return pl.concat(out,how='diagonal_relaxed')
def featlist(df,variant):
    f=feature_cols(df,GROUPS)+[c for c in df.columns if c.startswith(('v2_','x_','r_','s_'))]
    if 'notime' in variant: f=[c for c in f if c not in TIME]
    return f
def weights(df,variant):
    if 'wkw' not in variant: return None
    dow=((df['ts'].to_numpy()//86400)+3)%7   # 0=Mon (1970-01-01 was Thu)
    return np.where(dow>=5,0.3,1.0)
PARAMS=dict(objective='binary',learning_rate=0.05,num_leaves=63,min_data_in_leaf=200,feature_fraction=0.7,bagging_fraction=0.8,
            bagging_freq=1,lambda_l2=10.0,verbose=-1,seed=42,num_threads=8)
mode=sys.argv[1]
if mode=='val':
    cut=int(sys.argv[2]); variant=sys.argv[3] if len(sys.argv)>3 else 'base'
    import gc
    feats=None; Xs=[]; ys=[]
    for k in range(1,cut-20):
        t=load([f'e{k}' if EFT else f'f{k}'])
        if feats is None: feats=featlist(t,variant)
        Xs.append(t.select(feats).to_numpy().astype(np.float32)); ys.append(t['clicked'].to_numpy()); del t; gc.collect()
    X=np.concatenate(Xs); del Xs; gc.collect()
    ds=lgb.Dataset(X,np.concatenate(ys),feature_name=feats,free_raw_data=True).construct(); del X; gc.collect()
    m=lgb.train(PARAMS,ds,num_boost_round=450); del ds; gc.collect()
    va=load([f'val{cut}'])
    Xv=va.select(feats).to_numpy().astype(np.float32); y=va['clicked'].to_numpy()
    curve={k:roc_auc_score(y,m.predict(Xv,num_iteration=k)) for k in range(50,451,50)}
    best=max(curve,key=curve.get); p=m.predict(Xv,num_iteration=300)
    dow=((va['ts'].to_numpy()//86400)+3)%7
    cold=va['us_cold'].to_numpy()==1; noh=va['us_nohist'].to_numpy()==1
    segs={'all':np.ones(len(y),bool),'warm':~cold,'history_only':cold&~noh,'empty_history':noh,'weekday':dow<5,'weekend':dow>=5,'stale_lt1d':va['tm_stale'].to_numpy()<86400,'stale_ge1d':va['tm_stale'].to_numpy()>=86400}
    seg={k:(int(s.sum()),round(roc_auc_score(y[s],p[s]),5) if s.sum()>0 and y[s].min()<y[s].max() else None) for k,s in segs.items()}
    g=m.feature_importance('gain'); top=[(feats[i],round(float(g[i]/g.sum()),4)) for i in np.argsort(-g)[:20]]
    res=dict(cut=cut,variant=variant+('_eft' if EFT else '')+('_ne'),n_feats=len(feats),curve={k:round(v,5) for k,v in curve.items()},best=best,seg_at300=seg,top=top)
    print(json.dumps(res),flush=True)
    open('/workspace/huawei/v2work/combo_results.jsonl','a').write(json.dumps(res)+'\n')
    np.save(f'/workspace/huawei/v2work/combo_val{cut}_{variant}.npy',p)
elif mode=='final':
    import gc
    n_iter=int(sys.argv[2]); seeds=int(sys.argv[3]); variant=sys.argv[4] if len(sys.argv)>4 else 'base'
    feats=None; Xs=[]; ys=[]; ws=[]
    for k in range(1,7):   # load fold by fold -> float32 numpy to keep memory low
        t=load([f'E{k}' if EFT else f'f{k}'])
        if feats is None: feats=featlist(t,variant)
        Xs.append(t.select(feats).to_numpy().astype(np.float32)); ys.append(t['clicked'].to_numpy())
        w=weights(t,variant); ws.append(w if w is not None else np.ones(len(t)))
        del t; gc.collect()
    X=np.concatenate(Xs); del Xs; y=np.concatenate(ys); w=np.concatenate(ws) if 'wkw' in variant else None
    print('train rows',len(y),'feats',len(feats),flush=True)
    ds=lgb.Dataset(X,y,weight=w,feature_name=feats,free_raw_data=True).construct(); del X; gc.collect()
    preds=[]; ms=[]
    for s in range(seeds):
        m=lgb.train(dict(PARAMS,seed=42+s),ds,num_boost_round=n_iter)
        m.save_model(f'/workspace/huawei/v2work/{OUTP}_model_s{s}.txt'); ms.append(m); print('seed',s,'done',flush=True)
    del ds; gc.collect()
    ev=load(['eval']); Xe=ev.select(feats).to_numpy().astype(np.float32)
    p=np.mean([m.predict(Xe) for m in ms],0)   # plain probability average across seeds; no per-event normalisation
    pl.DataFrame({'rowid':ev['rowid'],'p':p}).write_parquet(f'/workspace/huawei/v2work/{OUTP}.parquet')
    g=m.feature_importance('gain'); print('top',[(feats[i],round(float(g[i]/g.sum()),4)) for i in np.argsort(-g)[:20]])
