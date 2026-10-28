"""resid.py: method (a) = DW residual boosting on top of v11 logits (init_score), features = DW reusable 42 cols + v11-derived context.
modes: oof (DW protocol: per cut 5-fold GroupKFold by user, seeds) ; xcut (train on the other val cuts' rows, users of the test cut removed -> predict test cut; mimics eval application)
       final (train on pooled val23/25/26 rows, predict eval; saves ens/resid_c27_s{seed}.npy as logit)"""
import sys, json, numpy as np, polars as pl, lightgbm as lgb, warnings; warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score as A
from sklearn.model_selection import GroupKFold
W='/workspace/huawei/v13'; V11='/workspace/huawei/v11'; F3='/workspace/huawei/v10/feat_v3'
PI=dict(objective='binary',learning_rate=0.03,num_leaves=15,min_data_in_leaf=500,feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,lambda_l2=30,max_bin=127,verbose=-1,num_threads=8)
def v11z(c):
    a=np.load(f'{V11}/ens/sCbig_c{c}_s0.npy')[:,1]; b=np.load(f'{V11}/ens/sC_s_c{c}_s0.npy')[:,1]
    m=np.mean([np.load(f'{V11}/ens/mlp2_c{c}_s{k}.npy')[:,3] for k in (range(3) if c!=26 else [0])],0); return ((a+b+2*m)/4).astype(np.float64)
def wev(x):
    m=x.reshape(-1,10); mu=m.mean(1,keepdims=True); sd=m.std(1,keepdims=True)+1e-6; return ((m-mu)/sd).ravel(),(m-mu).ravel(),m.argsort(1).argsort(1).astype(np.float32).ravel()
_HL=None
def hist_len(rowid):
    global _HL
    if _HL is None:
        L='/workspace/huawei/lite'
        te=pl.concat([pl.read_parquet(f'{L}/train_events.parquet',columns=['eid','h_items']),pl.read_parquet(f'{L}/evaluation_events.parquet',columns=['eid','h_items'])])
        rw=pl.concat([pl.read_parquet(f'{L}/train_rows.parquet',columns=['rowid','eid']),pl.read_parquet(f'{L}/evaluation_rows.parquet',columns=['rowid','eid'])])
        _HL=rw.join(te.select('eid',pl.col('h_items').list.len().alias('hl')),on='eid',how='left')
    return pl.DataFrame({'rowid':rowid}).join(_HL.select('rowid','hl'),on='rowid',how='left',maintain_order='left')['hl'].to_numpy().astype(np.float32)
def frame(c,use_dw=True):
    n='eval' if c==27 else f'val{c}'; m=pl.read_parquet(f'{F3}/{n}.parquet',columns=['rowid']+([] if c==27 else ['clicked']))
    z=v11z(c); assert len(z)==m.height; zz,dm,rk=wev(z); Z=z.reshape(-1,10); hl=hist_len(m['rowid'].to_numpy())
    B={'v11':z,'v11_z':zz,'v11_dm':dm,'v11_rk':rk,'ev_lse':np.repeat(np.log(np.exp(Z).sum(1)),10),'ev_max':np.repeat(Z.max(1),10),'ev_mean':np.repeat(Z.mean(1),10),'hist_len':hl,'nohist':(hl==0).astype(np.float32)}
    d=pl.read_parquet(f'{W}/feat_dw/{n}.parquet'); assert (d['rowid']==m['rowid']).all()
    Xb=np.column_stack(list(B.values())).astype(np.float32); Xd=d.drop('rowid').to_numpy().astype(np.float32)
    y=m['clicked'].to_numpy() if c!=27 else None
    user=pl.read_parquet(f'{F3}/{n}.parquet',columns=['u'])['u'].to_numpy() if 'u' in pl.read_parquet_schema(f'{F3}/{n}.parquet') else None
    return dict(Xb=Xb,Xd=Xd,y=y,z=z,user=user,cut=c)
def fit_pred(Xtr,ytr,itr,Xte,ite,seed,rounds=200):
    m=lgb.train({**PI,'seed':seed},lgb.Dataset(Xtr,ytr,init_score=itr),rounds); return m.predict(Xte,raw_score=True)+ite
if __name__=='__main__':
    mode=sys.argv[1]; seeds=[int(s) for s in sys.argv[2].split(',')]; cuts=[int(c) for c in sys.argv[3].split(',')] if len(sys.argv)>3 else [25,23,26]
    FR={c:frame(c) for c in set(cuts+[25,23,26])}
    for c in FR:
        if FR[c]['user'] is None: raise SystemExit('no user col')
    if mode=='oof':
        for c in cuts:
            f=FR[c]; y=f['y']; res={}
            for seed in seeds:
                p0=np.zeros(len(y)); p1=np.zeros(len(y))
                for tr,te in GroupKFold(5).split(f['Xb'],y,f['user']):
                    p0[te]=fit_pred(f['Xb'][tr],y[tr],f['z'][tr],f['Xb'][te],f['z'][te],seed)
                    X=np.hstack([f['Xb'],f['Xd']]); p1[te]=fit_pred(X[tr],y[tr],f['z'][tr],X[te],f['z'][te],seed)
                r=dict(mode='oof',cut=c,seed=seed,v11=A(y,f['z']),base=A(y,p0),new=A(y,p1)); r['d_vs_v11']=r['new']-r['v11']; r['d_vs_base']=r['new']-r['base']
                print(json.dumps({k:(round(v,5) if isinstance(v,float) else v) for k,v in r.items()}),flush=True); open(f'{W}/resid_results.jsonl','a').write(json.dumps(r)+'\n')
    elif mode=='xcut':
        for c in cuts:
            oth=[o for o in (25,23,26) if o!=c]; te=FR[c]; ute=set(te['user'].tolist())
            tr=[FR[o] for o in oth]; keep=[~np.isin(t['user'],list(ute)) for t in tr]
            Xtr=np.vstack([np.hstack([t['Xb'],t['Xd']])[k] for t,k in zip(tr,keep)]); ytr=np.concatenate([t['y'][k] for t,k in zip(tr,keep)]); itr=np.concatenate([t['z'][k] for t,k in zip(tr,keep)])
            Xtr_b=Xtr[:,:te['Xb'].shape[1]]; Xte=np.hstack([te['Xb'],te['Xd']])
            for seed in seeds:
                p0=fit_pred(Xtr_b,ytr,itr,te['Xb'],te['z'],seed); p1=fit_pred(Xtr,ytr,itr,Xte,te['z'],seed)
                r=dict(mode='xcut',cut=c,seed=seed,ntr=int(len(ytr)),v11=A(te['y'],te['z']),base=A(te['y'],p0),new=A(te['y'],p1)); r['d_vs_v11']=r['new']-r['v11']; r['d_vs_base']=r['new']-r['base']
                print(json.dumps({k:(round(v,5) if isinstance(v,float) else v) for k,v in r.items()}),flush=True); open(f'{W}/resid_results.jsonl','a').write(json.dumps(r)+'\n')
    elif mode=='final':
        te=frame(27); tr=[FR[o] for o in (25,23,26)]
        Xtr=np.vstack([np.hstack([t['Xb'],t['Xd']]) for t in tr]); ytr=np.concatenate([t['y'] for t in tr]); itr=np.concatenate([t['z'] for t in tr]); Xte=np.hstack([te['Xb'],te['Xd']])
        P=np.mean([fit_pred(Xtr,ytr,itr,Xte,te['z'],s) for s in seeds],0); np.save(f'{W}/ens/resid_c27.npy',P.astype(np.float32)); print('saved',P.shape)
