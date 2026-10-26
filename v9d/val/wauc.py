import numpy as np, polars as pl, json
from sklearn.metrics import roc_auc_score as A
V8='/workspace/huawei/v8'
for c in (25,26,23):
    v=pl.read_parquet(f'feat_v3/val{c}.parquet',columns=['rowid','clicked','eid','ts','tm_stale']).join(pl.read_parquet(f'{V8}/feat_sqv/all500/val{c}.parquet',columns=['rowid','sq_logit']),on='rowid',how='left',maintain_order='left')
    y=v['clicked'].to_numpy(); sq=v['sq_logit'].to_numpy(); w=np.load(f'ens/w_advni_val{c}.npy')
    zt=(np.load(f'ens/trim_b_c{c}_s0.npy')[:,1]+np.load(f'ens/trim_s_c{c}_s0.npy')[:,2])/2; z8=.8*zt+.2*sq
    zpp=np.load(f'{V8}/ens/pp_blend_c{c}.npy'); 
    r=dict(cut=c, z=A(y,zt), z8=A(y,z8), seq_gain=A(y,z8)-A(y,zt), sq_alone=A(y,sq), z_w=A(y,zt,sample_weight=w), z8_w=A(y,z8,sample_weight=w), seq_gain_w=A(y,z8,sample_weight=w)-A(y,zt,sample_weight=w))
    st=v['tm_stale'].to_numpy(); 
    for lo,hi in [(0,3600*6),(3600*6,86400),(86400,3*86400),(3*86400,1e9)]:
        m=(st>=lo)&(st<hi); 
        if m.sum()>1000 and 0<y[m].mean()<1: r[f'st{int(lo/3600)}h_n']=int(m.sum()); r[f'st{int(lo/3600)}h_z']=A(y[m],zt[m]); r[f'st{int(lo/3600)}h_gain']=A(y[m],z8[m])-A(y[m],zt[m])
    # days after cutoff
    C=1737331200+(c-20)*86400; d=((v['ts'].to_numpy()-C)//86400)
    for k in range(4):
        m=d==k
        if m.sum()>1000: r[f'd{k}_n']=int(m.sum()); r[f'd{k}_z']=A(y[m],zt[m]); r[f'd{k}_gain']=A(y[m],z8[m])-A(y[m],zt[m])
    print(json.dumps({k:(round(x,5) if isinstance(x,float) else x) for k,x in r.items()}))
