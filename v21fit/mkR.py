import numpy as np,polars as pl,zipfile,hashlib,os
from lib import *
from scipy.stats import spearmanr
A=np.load('A_all.npy'); D=np.load('D_dcn.npy'); V8=np.load('inv_v8regen.npy'); V5=np.load('inv_v5.npy'); mrow=M['rowid'].to_list()
Q1=EN-.136*E0+.098*(.5*A+.5*D)
nm='probe_v21fit_R1_Q1_m030v8regen_m015v5'
r=nr(Q1-.03*V8-.015*V5); assert np.isfinite(r).all() and len(r)==529390 and r.min()>=0 and r.max()<=1
csv=f'out/{nm}.csv'; pl.DataFrame({'rowid':mrow,'predicted_probability':r}).write_csv(csv,float_precision=12)
zp=csv+'.zip'; zf=zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED); zf.write(csv,arcname=os.path.basename(csv)); zf.close(); os.remove(csv)
h=hashlib.sha256(open(zp,'rb').read()).hexdigest()+'  '+os.path.basename(zp); open('out/SHA256SUMS_v21fit.txt','a').write(h+'\n'); print(h)
print('rho Q1',round(spearmanr(r[::3],nr(Q1)[::3])[0],5),'rho EN',round(spearmanr(r[::3],EN[::3])[0],5))
z=pl.read_csv(zipfile.ZipFile(zp).open(nm+'.csv').read()); print(z.shape,z.columns,z['rowid'].to_list()==mrow,z['predicted_probability'].null_count(),os.path.getsize(zp))
