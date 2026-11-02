import numpy as np,polars as pl,zipfile,hashlib,os
from lib import *
from scipy.stats import spearmanr
A=np.load('A_all.npy'); D=np.load('D_dcn.npy'); V8=np.load('inv_v8regen.npy'); V5=np.load('inv_v5.npy'); mrow=M['rowid'].to_list()
def f(p): c,g,q,v8,v5=p; return EN-c*E0+g*(1-q)*A+g*q*D+v8*V8+v5*V5
R1=nr(f([.136,.098,.5,-.03,-.015]))
P={'probe_v21fit_S1_R1step_c109_g123_q67_v8m052_v5m033':np.load('pS1.npy'),'probe_v21fit_S2_R1step_c097_g129_q75_v8m079_v5m015':np.load('pS2.npy')}
res={}; lines=[]
for nm,p in P.items():
    r=nr(f(p)); res[nm]=r; assert np.isfinite(r).all() and len(r)==529390 and r.min()>=0 and r.max()<=1
    csv=f'out/{nm}.csv'; pl.DataFrame({'rowid':mrow,'predicted_probability':r}).write_csv(csv,float_precision=12)
    zp=csv+'.zip'; zf=zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED); zf.write(csv,arcname=os.path.basename(csv)); zf.close(); os.remove(csv)
    lines.append(hashlib.sha256(open(zp,'rb').read()).hexdigest()+'  '+os.path.basename(zp))
    z=pl.read_csv(zipfile.ZipFile(zp).open(nm+'.csv').read()); print(nm,np.round(p,3),'rho R1',round(spearmanr(r[::3],R1[::3])[0],5),'rho EN',round(spearmanr(r[::3],EN[::3])[0],5),z.shape,z.columns,z['rowid'].to_list()==mrow,z['predicted_probability'].null_count())
k=list(res); print('rho S1 S2',round(spearmanr(res[k[0]][::3],res[k[1]][::3])[0],5))
open('out/SHA256SUMS_v21fit.txt','a').write('\n'.join(lines)+'\n'); print('\n'.join(lines))
