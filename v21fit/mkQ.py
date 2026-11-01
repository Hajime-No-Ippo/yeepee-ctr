import numpy as np,polars as pl,zipfile,hashlib,os
from lib import *
from scipy.stats import spearmanr
A=np.load('A_all.npy'); D=np.load('D_dcn.npy'); mrow=M['rowid'].to_list()
c={'probe_v21fit_Q1_EBnew_m136E0_p098_A50_DCN50':EN-.136*E0+.098*(.5*A+.5*D),
   'probe_v21fit_Q2_EBnew_m136E0_p098DCN':EN-.136*E0+.098*D}
lines=[]
for nm,x in c.items():
    r=nr(x); assert np.isfinite(r).all() and r.min()>=0 and r.max()<=1 and len(r)==529390
    csv=f'out/{nm}.csv'; pl.DataFrame({'rowid':mrow,'predicted_probability':r}).write_csv(csv,float_precision=12)
    zp=csv+'.zip'; zf=zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED); zf.write(csv,arcname=os.path.basename(csv)); zf.close(); os.remove(csv)
    lines.append(hashlib.sha256(open(zp,'rb').read()).hexdigest()+'  '+os.path.basename(zp)); print(nm,'rho EN',round(spearmanr(r[::3],EN[::3])[0],5),'rho P1',round(spearmanr(r[::3],nr(EN-.136*E0+.098*A)[::3])[0],5))
open('out/SHA256SUMS_v21fit.txt','a').write('\n'.join(lines)+'\n'); print('\n'.join(lines))
z=pl.read_csv(zipfile.ZipFile('out/probe_v21fit_Q1_EBnew_m136E0_p098_A50_DCN50.csv.zip').open('probe_v21fit_Q1_EBnew_m136E0_p098_A50_DCN50.csv').read()); print(z.shape,z.columns,z['rowid'].to_list()==mrow)
