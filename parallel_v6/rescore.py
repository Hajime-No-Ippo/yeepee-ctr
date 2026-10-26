"""Recompute v6 val AUCs from saved v2work preds (read-only). Row order = cl.matrix(valN, extra=('feat_lr2',)) with XDIR=feat_xf (17:53 cl.py key)."""
import os, sys, json, hashlib, numpy as np, polars as pl
sys.path.insert(0,'/workspace/huawei/v4')
from common import report
from sklearn.metrics import roc_auc_score
D='/workspace/huawei/v2work'; P3=f'{D}/feat_v3'
out={}
for cut in (25,26):
    n=f'val{cut}'
    key=hashlib.md5(json.dumps([n,1.0,'notime',['feat_lr2'],os.path.getmtime(f'{P3}/{n}.parquet')]+['feat_xf']).encode()).hexdigest()[:12]
    fp=f'{D}/cache_mx/{n}_{key}.parquet'; print(fp, os.path.exists(fp), flush=True)
    fl=json.load(open(fp+'.json'))
    t=pl.read_parquet(fp)
    b=[np.load(f'{D}/ens/v6base_c{cut}_s{s}.npy') for s in (0,1)]; sl=np.load(f'{D}/ens/v6slow_c{cut}_s0.npy')
    assert len(sl)==t.height
    y=t['clicked'].to_numpy()
    base=np.mean([x[:,1] for x in b],0); slow=sl[:,2]; z=(base+slow)/2
    r={'cache':os.path.basename(fp),'mtime':os.path.getmtime(fp),'rows':t.height,'n_feats':len(fl),'pos_rate':float(y.mean()),
       'auc_base_s0_400':roc_auc_score(y,b[0][:,1]),'auc_base_s1_400':roc_auc_score(y,b[1][:,1]),'auc_base_mean':roc_auc_score(y,base),
       'auc_slow_500':roc_auc_score(y,slow),'auc_blend':roc_auc_score(y,z)}
    # sensitivity of blend weight / rounds choice (selection-bias check)
    r['blend_weight_grid']={w:round(roc_auc_score(y,w*base+(1-w)*slow),5) for w in (0,0.25,0.5,0.75,1)}
    r['blend_rounds_grid']={f'b{rb}_s{rs}':round(roc_auc_score(y,(np.mean([x[:,i] for x in b],0)+sl[:,j])/2),5) for i,rb in enumerate((300,400)) for j,rs in enumerate((300,400,500))}
    m=t.select([c for c in ('rowid','eid','clicked','tm_stale') if c in t.columns])
    r['report_blend']=report(m.rename({'eid':'eid'}), 1/(1+np.exp(-z)))
    # compare row set & labels with our v3 split
    v3=pl.read_parquet(f'/workspace/huawei/v3/feat/{n}.parquet',columns=['rowid','clicked'])
    j=m.join(v3.rename({'clicked':'c3'}),on='rowid',how='inner')
    r['rows_common_with_v3split']=j.height; r['only_v6']=m.height-j.height; r['only_v3']=v3.height-j.height; r['label_mismatch']=int((j['clicked']!=j['c3']).sum())
    r['lr_feats']=[c for c in fl if c.startswith('lr')][:5]+['...',len([c for c in fl if c.startswith('lr')])]
    out[n]=r; print(json.dumps(r,indent=1,default=float),flush=True)
    np.save(f'/workspace/huawei/parallel_v6/v6_blend_{n}.npy',z)
    m.write_parquet(f'/workspace/huawei/parallel_v6/meta_{n}.parquet')
json.dump(out,open('/workspace/huawei/parallel_v6/score_val.json','w'),indent=1,default=float)
