"""Residual harness test: LightGBM on top of base logits (init_score), GroupKFold by user.  usage: run_trend.py CUT [groups,comma] [seeds]
control = base-context cols only; test = + group cols; noise = same cols, rows permuted across events (distribution preserved, no signal)."""
import sys; sys.dont_write_bytecode=True
sys.path.insert(0,'/workspace/huawei/dw_lb'); sys.path.insert(0,'/workspace/huawei/dw_trend')
import lib; from lib import *
lib.PI['num_threads']=2
from base import base_logit
CACHE='/workspace/huawei/dw_trend/cache'
ITEM=['it_tr_ratio','it_tr_slope','it_imp_gr','it_clk_gr','it_hrs_clk']; USER=['u_rate_tr','u_gap_tr','u_overdue']; TASTE=['tx_cos_rec','tx_cos_old','tx_dcos','tx_drift','tx_proj']
TT=['TT_imp_ratio','TT_imp_loc','TT_slate_pop','TT_coimp_mean','TT_coimp_max']
DMC=TT[:2]+TT[3:]+ITEM+['tx_cos_rec','tx_cos_old','tx_dcos','tx_proj']
def load(c):
    m=meta(c); B,z,M=base_logit(c); df=pl.read_parquet(f'{CACHE}/trend_c{c}.parquet')
    import os
    if os.path.exists(f'{CACHE}/tt_c{c}.parquet'): df=df.join(pl.read_parquet(f'{CACHE}/tt_c{c}.parquet'),on='rowid',how='left',maintain_order='left')
    assert (df['rowid'].to_numpy()==m['rowid'].to_numpy()).all()
    F=dict(c=c,y=m['clicked'].to_numpy().astype(int),z=B,user=m['u'].to_numpy(),Xb=base_ctx(B)); return F,df
def cols(df,names,dm=True):
    X=[df[n].to_numpy().astype(np.float32) for n in names]
    if dm:
        for n in names:
            if n in DMC: a=df[n].to_numpy().astype(np.float64).reshape(-1,10); X.append((a-np.nanmean(a,1,keepdims=True)).ravel().astype(np.float32))
    return np.column_stack(X)
def run(F,df,name,names,seeds,nperm=1,log='results_trend.jsonl'):
    y,z,u=F['y'],F['z'],F['user']; X=cols(df,names); n=len(y)//10
    pc=np.mean([oof(F['Xb'],y,z,u,s) for s in seeds],0); pt=np.mean([oof(np.hstack([F['Xb'],X]),y,z,u,s) for s in seeds],0)
    r=dict(name=name,cut=F['c'],nfeat=X.shape[1],base=A(y,z),control=A(y,pc),test=A(y,pt)); r['d_vs_ctrl']=r['test']-r['control']; r['d_vs_base']=r['test']-r['base']
    ci,sd=boot_delta(y,pc,pt,None); r['ci']=[float(ci[0]),float(ci[1])]; r['sd']=float(sd)
    nz=[]
    for k in range(nperm):
        perm=np.random.RandomState(100+k).permutation(n); Xn=X.reshape(n,10,-1)[perm].reshape(n*10,-1)
        pn=np.mean([oof(np.hstack([F['Xb'],Xn]),y,z,u,s) for s in seeds],0); nz.append(A(y,pn)-r['control'])
    r['noise_d']=nz
    print(json.dumps({k:(round(v,5) if isinstance(v,float) else ([round(x,5) for x in v]) if isinstance(v,list) else v) for k,v in r.items()}),flush=True)
    open(f'/workspace/huawei/dw_trend/{log}','a').write(json.dumps(r)+'\n'); return r
GROUPS=dict(item=ITEM,user=USER,taste=TASTE,all=ITEM+USER+TASTE,TT_group=TT,TT_plus_all=TT+ITEM+USER+TASTE)
if __name__=='__main__':
    c=int(sys.argv[1]); gs=sys.argv[2].split(',') if len(sys.argv)>2 else list(GROUPS); seeds=[int(x) for x in sys.argv[3].split(',')] if len(sys.argv)>3 else [0,1]
    F,df=load(c)
    for g in gs: run(F,df,g,GROUPS[g] if g in GROUPS else g.split('+'),seeds)
