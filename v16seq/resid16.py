"""resid16.py MODE FEATSPEC [seeds] : DW-style residual boosting on v12 logits (init_score), base ctx features vs base+seq features.
FEATSPEC e.g. 'gru_s0+gru_s1,sas_s0' : comma = separate models; '+' = average of seeds columns. MODE oof (GroupKFold5 by user) | xcut (train on other val cuts' rows, users of test cut removed)"""
import sys,json,numpy as np,polars as pl,lightgbm as lgb,warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'/workspace/huawei/v12'); sys.path.insert(0,'/workspace/huawei/v12x')
from resid import PI,fit_pred,wev,hist_len
from sklearn.metrics import roc_auc_score as A
from sklearn.model_selection import GroupKFold
F3='/workspace/huawei/v12/feat_v3'; V='/workspace/huawei/v12/ens'; FD='/workspace/huawei/v16seq/feat'
def ld(p,col): return np.load(p)[:,col].astype(np.float64)
def v12z(c):
    big=ld(f'{V}/dwbig_c{c}_s0.npy',1); sm=ld(f'{V}/dw_s_c{c}_s0.npy',1); ml=np.mean([ld(f'{V}/dwmlp_c{c}_s{s}.npy',3) for s in (range(3) if c!=26 else [0])],0); return (big+sm+2*ml)/4
def seqf(spec,n,rowid):
    outs=[];names=[]
    for grp in spec.split(','):
        dfs=[pl.read_parquet(f'{FD}/{d}/{n}.parquet') for d in grp.split('+')]; assert all((d['rowid'].to_numpy()==rowid).all() for d in dfs)
        cols=[c for c in dfs[0].columns if c!='rowid']; outs.append(np.nanmean(np.stack([d.select(cols).to_numpy() for d in dfs]),0)); names+=cols if len(spec.split(','))==1 else [f'{grp}:{c}' for c in cols]
    return np.hstack(outs).astype(np.float32),names
def frame(c,spec):
    n=f'val{c}'; m=pl.read_parquet(f'{F3}/{n}.parquet',columns=['rowid','clicked','u']); z=v12z(c); zz,dm,rk=wev(z); Z=z.reshape(-1,10); hl=hist_len(m['rowid'].to_numpy())
    Xb=np.column_stack([z,zz,dm,rk,np.repeat(np.log(np.exp(Z).sum(1)),10),np.repeat(Z.max(1),10),np.repeat(Z.mean(1),10),hl,(hl==0)]).astype(np.float32)
    Xs,nm=seqf(spec,n,m['rowid'].to_numpy()); return dict(Xb=Xb,Xs=Xs,y=m['clicked'].to_numpy(),z=z,user=m['u'].to_numpy(),cut=c,names=nm)
if __name__=='__main__':
    mode,spec=sys.argv[1],sys.argv[2]; seeds=[int(s) for s in sys.argv[3].split(',')] if len(sys.argv)>3 else [0,1]; cuts=[25,23,26]
    FR={c:frame(c,spec) for c in cuts}
    for c in cuts:
        f=FR[c]; y=f['y']
        if mode=='oof':
            for seed in seeds:
                p0=np.zeros(len(y));p1=np.zeros(len(y)); X=np.hstack([f['Xb'],f['Xs']])
                for tr,te in GroupKFold(5).split(X,y,f['user']):
                    p0[te]=fit_pred(f['Xb'][tr],y[tr],f['z'][tr],f['Xb'][te],f['z'][te],seed); p1[te]=fit_pred(X[tr],y[tr],f['z'][tr],X[te],f['z'][te],seed)
                r=dict(mode='oof',spec=spec,cut=c,seed=seed,v12=A(y,f['z']),base=A(y,p0),new=A(y,p1)); r['d_vs_v12']=r['new']-r['v12']; r['d_vs_base']=r['new']-r['base']
                print(json.dumps({k:(round(v,5) if isinstance(v,float) else v) for k,v in r.items()}),flush=True); open('/workspace/huawei/v16seq/resid_results.jsonl','a').write(json.dumps(r)+'\n')
        else:
            oth=[o for o in cuts if o!=c]; ute=np.unique(f['user']); tr=[FR[o] for o in oth]; keep=[~np.isin(t['user'],ute) for t in tr]
            Xtr=np.vstack([np.hstack([t['Xb'],t['Xs']])[k] for t,k in zip(tr,keep)]); ytr=np.concatenate([t['y'][k] for t,k in zip(tr,keep)]); itr=np.concatenate([t['z'][k] for t,k in zip(tr,keep)]); nb=f['Xb'].shape[1]
            for seed in seeds:
                p0=fit_pred(Xtr[:,:nb],ytr,itr,f['Xb'],f['z'],seed); p1=fit_pred(Xtr,ytr,itr,np.hstack([f['Xb'],f['Xs']]),f['z'],seed)
                r=dict(mode='xcut',spec=spec,cut=c,seed=seed,ntr=int(len(ytr)),v12=A(y,f['z']),base=A(y,p0),new=A(y,p1)); r['d_vs_v12']=r['new']-r['v12']; r['d_vs_base']=r['new']-r['base']
                print(json.dumps({k:(round(v,5) if isinstance(v,float) else v) for k,v in r.items()}),flush=True); open('/workspace/huawei/v16seq/resid_results.jsonl','a').write(json.dumps(r)+'\n')
