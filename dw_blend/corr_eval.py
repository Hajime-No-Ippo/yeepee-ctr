import os
os.environ['OMP_NUM_THREADS']='1'
import numpy as np, glob, polars as pl
from scipy.stats import rankdata
H='/workspace/huawei/'
nr=lambda x:(rankdata(x)-1)/(len(x)-1)
M=pl.read_parquet(H+'erbrown/merged.parquet')
A={'EN(EB-new)':np.load(H+'erbrown/w10an/eb_new_rank.npy'),'E0(submission_best)':nr(M['eb'].to_numpy()),'ours_O(v21list B)':nr(M['ours'].to_numpy()),'p33':nr(M['p33_rank'].to_numpy())}
for k,f in [('A_all(ctl6+big)','A_all.npy'),('B_all(TRANSD)','B_all.npy'),('D_dcn','D_dcn.npy'),('S_ebdistill','S_ebdistill.npy'),('S1(0.829101)','pS1.npy'),('S2','pS2.npy')]:
    A[k]=nr(np.load(H+'v21fit/'+f))
for f in sorted(glob.glob(H+'v21fit/inv_*.npy')): A[os.path.basename(f)[4:-4]]=np.load(f)
A={k:v for k,v in A.items() if v.shape==(529390,)}
names=list(A); X=np.column_stack([A[k][::3] for k in names]); C=np.corrcoef(X.T)
df=pl.DataFrame({'model':names,**{n:C[:,i].round(4) for i,n in enumerate(names)}}); df.write_csv('corr_eval.csv')
pl.Config.set_tbl_cols(30); pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(250)
print(df.select(['model','EN(EB-new)','E0(submission_best)','A_all(ctl6+big)','D_dcn','p33']))
