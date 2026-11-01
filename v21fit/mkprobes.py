import numpy as np,polars as pl,zipfile,hashlib,os
from lib import *
A=np.load('A_all.npy'); S=np.load('S_ebdistill.npy')
mrow=M['rowid'].to_list()
cands={
 'probe_v21fit_P1_EBnew_m136E0_p098A':EN-.136*E0+.098*A,
 'probe_v21fit_P2_EBDISTILL_EBnew90_S10':.9*EN+.1*S,
 'probe_v21fit_P3_EBDISTILL_EBnew_m136E0_p050A_p050S':EN-.136*E0+.05*A+.05*S,
}
sums=[]
from scipy.stats import spearmanr
for nm,x in cands.items():
    r=nr(x); assert np.isfinite(r).all() and r.min()>=0 and r.max()<=1 and len(r)==529390
    csv=f'out/{nm}.csv'; pl.DataFrame({'rowid':mrow,'predicted_probability':r}).write_csv(csv,float_precision=12)
    zp=csv+'.zip'; zf=zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED); zf.write(csv,arcname=os.path.basename(csv)); zf.close(); os.remove(csv)
    sums.append(hashlib.sha256(open(zp,'rb').read()).hexdigest()+'  '+os.path.basename(zp))
    print(nm,'rho vs EN',round(spearmanr(r[::3],EN[::3])[0],5))
open('out/SHA256SUMS_v21fit.txt','w').write('\n'.join(sums)+'\n'); print('\n'.join(sums))
# format check vs previous submission
z=zipfile.ZipFile('/workspace/huawei/erbrown/w10an/submission_blend_newbest_v21_w10_20261031.csv.zip'); print(z.read(z.namelist()[0])[:80])
z=zipfile.ZipFile('out/probe_v21fit_P1_EBnew_m136E0_p098A.csv.zip'); print(z.read(z.namelist()[0])[:80]); 
