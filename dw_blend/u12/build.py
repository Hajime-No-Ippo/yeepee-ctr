import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import numpy as np, polars as pl, zipfile, hashlib, csv, io
from scipy.stats import rankdata
V='/workspace/huawei/v21fit/'
M=pl.read_parquet('/workspace/huawei/erbrown/merged.parquet',columns=['rowid','eb']); rid=M['rowid'].to_list()
print('rowid sha',hashlib.sha256('\n'.join(map(str,rid)).encode()).hexdigest())
nr=lambda x:(rankdata(x)-1)/(len(x)-1)
EN=np.load('/workspace/huawei/erbrown/w10an/eb_new_rank.npy'); E0=nr(M['eb'].to_numpy())
A=np.load(V+'A_all.npy'); D=np.load(V+'D_dcn.npy'); V8=np.load(V+'inv_v8regen.npy'); V5=np.load(V+'inv_v5.npy')
def blend(p):
    c,g,q,v8,v5=map(np.float64,p); return EN-c*E0+g*(1-q)*A+g*q*D+v8*V8+v5*V5
def rz(p):
    z=zipfile.ZipFile(p); return pl.read_csv(io.BytesIO(z.read(z.namelist()[0])))['predicted_probability'].to_numpy()
# sanity: reproduce S1 and T1 from files
S1=nr(blend(np.load(V+'pS1.npy'))); s1f=rz(V+'out/probe_v21fit_S1_R1step_c109_g123_q67_v8m052_v5m033.csv.zip')
print('S1 rebuild max|diff| vs file',np.abs(S1-s1f).max())
T=nr(blend((.081,.148,.834,-.075,-.052)))
R={'S1':S1,'T1':rz(V+'out/probe_v21fit_T1_a1_R1step067_c081_g148_q83_v8m075_v5m052.csv.zip'),'R1':rz(V+'out/probe_v21fit_R1_Q1_m030v8regen_m015v5.csv.zip')}
P={'U1':('probe_v21fit_U1_c090_g160_q100_v8m090_v5m040',(.09,.16,1.0,-.09,-.04)),'U2':('probe_v21fit_U2_c090_g160_q090_v8m090_v5m040',(.09,.16,.9,-.09,-.04))}
os.makedirs('out',exist_ok=True)
for k,(nm,p) in P.items():
    r=nr(blend(p)); assert np.isfinite(r).all() and r.min()>=0 and r.max()<=1 and len(r)==529390
    np.save(f'{k}.npy',r); R[k]=r
    cp=f'out/{nm}.csv'
    with open(cp,'w',newline='') as f:
        w=csv.writer(f,lineterminator='\n'); w.writerow(['rowid','predicted_probability'])
        for i,x in zip(rid,r): w.writerow([i,format(x,'.12f')])
    csvsha=hashlib.sha256(open(cp,'rb').read()).hexdigest()
    zp=cp+'.zip'
    with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED) as zf: zf.write(cp,arcname=os.path.basename(cp))
    print(k,'csv sha',csvsha,'zip sha',hashlib.sha256(open(zp,'rb').read()).hexdigest())
np.savez('arrs.npz',**R,EN=EN,E0=E0,A=A,D=D)
