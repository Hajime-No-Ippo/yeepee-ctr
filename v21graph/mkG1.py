import sys,os,zipfile,hashlib; sys.dont_write_bytecode=True
os.chdir('/workspace/huawei/v21fit'); sys.path.insert(0,'/workspace/huawei/v21fit')
import numpy as np,polars as pl
from lib import *
from scipy.stats import spearmanr
A=np.load('A_all.npy'); D=np.load('D_dcn.npy'); V8=np.load('inv_v8regen.npy'); V5=np.load('inv_v5.npy'); mrow=M['rowid'].to_list()
c,g,q,v8,v5=np.load('pS1.npy'); S1=nr(EN-c*E0+g*(1-q)*A+g*q*D+v8*V8+v5*V5)
G=np.load('/workspace/huawei/v21graph/out/graph_eval_v21seqorder.npy').astype(np.float64); rid=np.load('/workspace/huawei/v21graph/out/graph_eval_rowid.npy').astype(str)
gm=pl.DataFrame({'rowid':rid,'p':G}).join(pl.DataFrame({'rowid':mrow}),on='rowid',how='right',maintain_order='right')['p'].to_numpy(); assert np.isfinite(gm).all()
Gr=nr(gm); np.save('/workspace/huawei/v21graph/out/graph_eval_merged_rank.npy',Gr.astype(np.float32))
r=nr(S1-0.045*Gr); assert np.isfinite(r).all() and len(r)==529390
nm='probe_v21graph_G1_S1_m045graph7s'; out='/workspace/huawei/v21graph/out/'; csv=out+nm+'.csv'
pl.DataFrame({'rowid':mrow,'predicted_probability':r}).write_csv(csv,float_precision=12)
zp=csv+'.zip'; zf=zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED); zf.write(csv,arcname=nm+'.csv'); zf.close(); os.remove(csv)
s1=rz('out/probe_v21fit_S1_R1step_c109_g123_q67_v8m052_v5m033.csv.zip')
z=pl.read_csv(zipfile.ZipFile(zp).open(nm+'.csv').read())
print('rows',z.shape,z.columns,'ids==S1',z['rowid'].to_list()==mrow,'nulls',z['predicted_probability'].null_count(),'min/max',z['predicted_probability'].min(),z['predicted_probability'].max())
print('rho vs S1 file',spearmanr(z['predicted_probability'].to_numpy()[::2],s1[::2])[0],'rebuilt S1 vs S1 file',spearmanr(S1[::5],s1[::5])[0],'rho graph vs S1',spearmanr(Gr[::5],S1[::5])[0])
h=hashlib.sha256(open(zp,'rb').read()).hexdigest(); print(h,os.path.getsize(zp)); open(out+'SHA256SUMS_v21graph.txt','a').write(h+'  '+nm+'.csv.zip\n')
