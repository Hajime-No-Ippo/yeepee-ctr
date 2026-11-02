import sys; sys.path.insert(0,'../evclick')
from evlib import *
import lightgbm as lgb, time
from sklearn.metrics import roc_auc_score
B='/workspace/huawei/cb_blend/'
r=json.load(open(H+'/v20/gain_rank.json'))['ranked']; F80=[x['feature'] for x in r[:80]]
TR={23:['f1','f2'],25:['f1','f2','f3','f4'],26:['f2','f3','f4','f5']}
P=dict(objective='binary',learning_rate=.05,num_leaves=63,min_data_in_leaf=200,feature_fraction=.8,bagging_fraction=.8,bagging_freq=1,lambda_l2=10,max_bin=127,num_threads=4,verbose=-1,seed=0)
def get(ns,extra=()):
    ds=[]
    for n in ns:
        d,miss=cols(n,F80); assert not miss,miss; ds.append(d.select(['rowid','eid','clicked','tm_stale']+F80).with_columns([pl.col(c).cast(pl.Float32) for c in F80]))
    return pl.concat(ds)
for cut in [int(a) for a in sys.argv[1:]] or (23,25):
    tr=get(TR[cut]); va=get(f'val{cut}'.split()) if False else get([f'val{cut}'])
    o=np.load(B+f'oof_c{cut}.npz',allow_pickle=True); rid=o['rowid']; y=o['y']
    pos=pl.DataFrame(dict(rowid=rid,i=np.arange(len(rid)))); 
    va=va.join(pos,on='rowid',how='left').sort('i'); assert (va['rowid'].to_numpy()==rid).all() and (va['clicked'].to_numpy()==y).all()
    t=va['tm_stale'].to_numpy(); ttr=tr['tm_stale'].to_numpy()
    ytr=tr['clicked'].to_numpy(); Xtr=tr.select(F80).to_numpy(); Xva=va.select(F80).to_numpy()
    Xtr_t=np.c_[Xtr,ttr.astype(np.float32)]; Xva_t=np.c_[Xva,t.astype(np.float32)]
    V={'base80':(Xtr,None,Xva),'w':(Xtr,1+3*(ttr/86400.)**2,Xva),'wt':(Xtr_t,1+3*(ttr/86400.)**2,Xva_t),'t':(Xtr_t,None,Xva_t)}
    for k,(X,w,Xv) in V.items():
        t0=time.time(); m=lgb.train(P,lgb.Dataset(X,ytr,weight=w),400); p=m.predict(Xv)
        np.save(f'pred_{k}_c{cut}.npy',p.astype(np.float32))
        st=t>86400
        print(cut,k,'AUC all %.4f'%roc_auc_score(y,p),'fresh %.4f'%roc_auc_score(y[~st],p[~st]),('stale %.4f'%roc_auc_score(y[st],p[st])) if st.any() else '',round(time.time()-t0),'s',flush=True)
    del tr,va,Xtr,Xva,Xtr_t,Xva_t; gc.collect()
