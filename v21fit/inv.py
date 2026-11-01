import numpy as np,polars as pl,zipfile,io,glob,os
from lib import *
from scipy.stats import spearmanr
fs={'v21slateB':'v21diag/out/submission_v21slate_B_TRANSDUCTIVE.csv.zip','v21slateA':'v21diag/out/submission_v21slate_A_trainonly.csv.zip','v21listA':'v21diag/out/submission_v21list_A_trainonly.csv.zip',
'p3_d16':'teammate/submission_p3_d16.zip','p33blend70':'teammate/submission_p33_blend_w70.zip','probe_blend_p3':'teammate/probe_blend_p3.csv.zip','v8regen':'v12x/v8regen/submission_v8_regen.csv.zip','v5':'v5/submission_v5.csv.zip','v5c':'v5/submission_v5c.csv.zip','best_E0':'erbrown/submission_best.csv.zip','w10':'erbrown/w10an/submission_blend_newbest_v21_w10_20261031.csv.zip'}
R={}
for k,f in fs.items():
    try:
        z=zipfile.ZipFile(H+f); d=pl.read_csv(io.BytesIO(z.read(z.namelist()[0])))
        d=pl.DataFrame({'rowid':M['rowid']}).join(d,on='rowid',how='left',maintain_order='left'); p=d['predicted_probability'].to_numpy()
        assert np.isfinite(p).all(); R[k]=nr(p); np.save(f'inv_{k}.npy',R[k])
        print(k,'rho EN',round(spearmanr(R[k][::4],EN[::4])[0],4),'rho A',round(spearmanr(R[k][::4],np.load('A_all.npy')[::4])[0],4),'rho P',round(spearmanr(R[k][::4],P[::4])[0],4))
    except Exception as e: print(k,'ERR',str(e)[:100])
