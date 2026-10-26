"""Check v6 run used fresh caches (built after feat_xf at 18:49-18:51) and that cached cols equal on-disk tables."""
import os, json, hashlib, time, numpy as np, polars as pl
D='/workspace/huawei/v2work'; P3=f'{D}/feat_v3'; out={}
for n in ['f1','f2','f3','f4','f5','f6','val25','val26','eval']:
    key=hashlib.md5(json.dumps([n,1.0,'notime',['feat_lr2'],os.path.getmtime(f'{P3}/{n}.parquet')]+['feat_xf']).encode()).hexdigest()[:12]
    fp=f'{D}/cache_mx/{n}_{key}.parquet'
    r={'cache':os.path.basename(fp),'exists':os.path.exists(fp)}
    if os.path.exists(fp):
        r['mtime']=time.strftime('%H:%M:%S',time.localtime(os.path.getmtime(fp)))
        r['xf_mtime']=time.strftime('%H:%M:%S',time.localtime(os.path.getmtime(f'{D}/feat_xf/{n}.parquet')))
        cols=['rowid','x_icon05_k64','x_clctr64','x_con05_k32','lr8_cos','lr16_con05_z']
        c=pl.read_parquet(fp,columns=cols)
        s=pl.read_parquet(f'{D}/feat_xf/{n}.parquet',columns=['rowid','x_icon05_k64','x_clctr64','x_con05_k32']).join(
          pl.read_parquet(f'{D}/feat_lr2/{n}.parquet',columns=['rowid','lr8_cos','lr16_con05_z']),on='rowid')
        j=c.join(s,on='rowid',suffix='_src')
        r['rows']=c.height; r['joined']=j.height
        r['maxdiff']={k:float(np.nanmax(np.abs(j[k].cast(pl.Float64).to_numpy()-j[k+'_src'].cast(pl.Float32).cast(pl.Float64).to_numpy()))) for k in cols[1:]}
        # old (pre-fix) feat_x differs?
        if os.path.exists(f'{D}/feat_x/{n}.parquet'):
            o=pl.read_parquet(f'{D}/feat_x/{n}.parquet',columns=['rowid','x_icon05_k64']).join(c.select('rowid','x_icon05_k64'),on='rowid',suffix='_v6')
            r['corr_icon_old_vs_v6']=float(np.corrcoef(o['x_icon05_k64'].to_numpy(),o['x_icon05_k64_v6'].to_numpy())[0,1])
    out[n]=r; print(n,r,flush=True)
json.dump(out,open('cachecheck.json','w'),indent=1)
