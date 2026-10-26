import numpy as np, polars as pl, csv, math, hashlib, zipfile, os
O='/workspace/huawei/v2work/out'
m=pl.read_parquet('cache_mx/eval_93841df7c9f9.parquet',columns=['rowid'])
sub=pl.read_csv(f'{O}/submission_v8.csv',schema_overrides={'rowid':pl.Utf8})
sq=pl.read_parquet('../feat_sq/eval.parquet',columns=['rowid','sq_logit']).with_columns(pl.col('rowid').cast(pl.Utf8))
j=m.join(sub,on='rowid',how='left',maintain_order='left').join(sq,on='rowid',how='left',maintain_order='left')
p8=j['predicted_probability'].to_numpy().astype(float); L8=np.log(p8/(1-p8)); s=j['sq_logit'].to_numpy().astype(float); lr=np.load('ens/lr_c27.npy').astype(float)
sig=lambda x:1/(1+np.exp(-x))
P={'lr':sig(0.8*L8+0.2*lr),'sqonly':sig(s)}
ids=sub['rowid']  # submission_v8 order = evaluation.csv order (validated at ship)
for name,p in P.items():
    df=pl.DataFrame({'rowid':j['rowid'],'p':p}); df=pl.DataFrame({'rowid':ids}).join(df,on='rowid',how='left',maintain_order='left')
    assert df['p'].null_count()==0
    out=f'/tmp/probe_{name}.csv'
    with open(out,'w',newline='') as f:
        f.write('rowid,predicted_probability\n'); f.writelines(f'{r},{v:.8g}\n' for r,v in zip(df['rowid'].to_list(),df['p'].to_numpy()))
    rows=list(csv.reader(open(out))); h=rows[0]; rows=rows[1:]; v=np.array([float(x[1]) for x in rows])
    ok=h==['rowid','predicted_probability'] and len(rows)==529390 and len({x[0] for x in rows})==529390 and {x[0] for x in rows}==set(ids.to_list()) and np.isfinite(v).all() and (v>=0).all() and (v<=1).all()
    zp=f'{O}/probe_{name}.csv.zip'
    with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED) as z: z.write(out,arcname=f'probe_{name}.csv')
    with zipfile.ZipFile(zp) as z: nl=z.namelist(); zok=z.testzip() is None
    from scipy.stats import spearmanr
    print(name,'VALID' if ok and zok else 'INVALID',nl,'mean %.4f min %.3g max %.3g'%(v.mean(),v.min(),v.max()),'csv sha256',hashlib.sha256(open(out,'rb').read()).hexdigest(),'zip sha256',hashlib.sha256(open(zp,'rb').read()).hexdigest(),os.path.getsize(zp))
    from scipy.stats import spearmanr
    print('  spearman vs v8', round(spearmanr(p,p8).correlation,5))
    os.remove(out)
