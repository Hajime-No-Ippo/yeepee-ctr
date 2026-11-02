import numpy as np,polars as pl,zipfile,hashlib,os,pickle
from lib import *
from scipy.stats import spearmanr
A=np.load('A_all.npy'); D=np.load('D_dcn.npy'); V8=np.load('inv_v8regen.npy'); V5=np.load('inv_v5.npy'); mrow=M['rowid'].to_list()
def f(p): c,g,q,v8,v5=p; return EN-c*E0+g*(1-q)*A+g*q*D+v8*V8+v5*V5
pR=np.array([.136,.098,.5,-.03,-.015]); p0=pickle.load(open('reopt_out.pkl','rb'))[1]; pS2=np.load('pS2.npy'); q5=pR+2*(pS2-pR)
R1=nr(f(pR)); S1=nr(f(np.load('pS1.npy'))); S2=nr(f(pS2))
P={'probe_v21fit_T1_a1_R1step067_c081_g148_q83_v8m075_v5m052':pR+.667*(p0-pR),
   'probe_v21fit_T2_a2_R1fullstep_c054_g172_q100_v8m098_v5m070':p0,
   'probe_v21fit_T3_b_v5held_c077_g144_q88_v8m103_v5m015':pR+.75*(q5-pR),
   'probe_v21fit_T4_c_R1_q75_DCNshare_only':np.array([.136,.098,.75,-.03,-.015])}
lines=[];info=[]
for nm,p in P.items():
    r=nr(f(p)); assert np.isfinite(r).all() and len(r)==529390 and r.min()>=0 and r.max()<=1
    csv=f'out/{nm}.csv'; pl.DataFrame({'rowid':mrow,'predicted_probability':r}).write_csv(csv,float_precision=12)
    zp=csv+'.zip'; zf=zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED); zf.write(csv,arcname=os.path.basename(csv)); zf.close(); os.remove(csv)
    h=hashlib.sha256(open(zp,'rb').read()).hexdigest(); lines.append(h+'  '+os.path.basename(zp))
    z=pl.read_csv(zipfile.ZipFile(zp).open(nm+'.csv').read())
    print(nm,np.round(p,3),'rho R1 %.5f S1 %.5f S2 %.5f EN %.5f'%tuple(spearmanr(r[::3],x[::3])[0] for x in (R1,S1,S2,EN)),z.shape,z.columns,z['rowid'].to_list()==mrow,z['predicted_probability'].null_count(),os.path.getsize(zp))
open('out/SHA256SUMS_v21fit.txt','a').write('\n'.join(lines)+'\n'); print('\n'.join(lines))
