import os; os.environ['OMP_NUM_THREADS']='2'
import numpy as np, polars as pl, zipfile, hashlib
from scipy.special import expit
from scipy.stats import rankdata, spearmanr
H='/workspace/huawei'; O=f'{H}/v21diag/out'; os.makedirs(O,exist_ok=True); W_=0.6
ref=pl.read_csv(f'{H}/v5/submission_v5.csv',columns=['rowid'])['rowid'].to_list(); q=pl.read_parquet(f'{H}/teammate/work33/p33_recovered_ranks.parquet').select(['rowid','p33_rank'])
rid=pl.read_parquet(f'{H}/v12/feat_v3/eval.parquet',columns=['rowid'])['rowid'].to_list(); ev=pl.read_csv(f'{H}/data/evaluation.csv',columns=['rowid'])['rowid'].to_list()
assert set(ref)==set(ev)==set(rid)==set(q['rowid'].to_list()) and len(ref)==529390
def zipit(csv):
    zp=csv+'.zip'; zf=zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED); zf.write(csv,arcname=os.path.basename(csv)); zf.close()
    sh=hashlib.sha256(open(zp,'rb').read()).hexdigest(); open(zp+'.sha256','w').write(f'{sh}  {os.path.basename(csv)}.zip\n'); return sh
def ordered(vals,order): return pl.DataFrame({'rowid':rid,'p':vals}).join(pl.DataFrame({'rowid':order}),on='rowid',how='right',maintain_order='right')
for v,nm in (('B','B_TRANSDUCTIVE'),('A','A_trainonly')):
    L=lambda f,col:np.load(f'{H}/v19/ens/{f}.npy')[:,col]
    b=L(f'nm{v}s_c27_s0',0); d=L(f'nm{v}s_s_c27_s0',1); ml=np.mean([L(f'nm{v}smlp_c27_s{k}',3) for k in (0,1,2)],0); z=(b+d+2*ml)/4
    Ls=[np.load(f'ens/v21list{v}_c27_s{k}.npy')[:,3] for k in (0,1,2)]; lp=np.minimum(np.mean(Ls,0),-1e-6); Lg=lp-np.log1p(-np.exp(lp)); assert len(Lg)==len(z)
    print(' spearman v19 vs list',round(spearmanr(z[::5],Lg[::5])[0],4)); z2=(1-W_)*z+W_*Lg
    p_ref=ordered(expit(z),ev)['p'].to_numpy(); df=ordered(expit(z2),ev); p=df['p'].to_numpy()
    assert df['rowid'].to_list()==ev and np.isfinite(p).all() and 0<=p.min() and p.max()<=1
    name=f'submission_v21list_{nm}'; pl.DataFrame({'rowid':ev,'predicted_probability':p}).write_csv(f'{O}/{name}.csv'); print(name,zipit(f'{O}/{name}.csv'))
    print(' spearman vs v19 blend',round(spearmanr(p,p_ref)[0],5))
    qq=pl.DataFrame({'rowid':ev}).join(q,on='rowid',how='left',maintain_order='left')['p33_rank'].to_numpy(); n=len(p)
    sc=np.clip(0.8*(rankdata(p)-1)/(n-1)+0.2*(rankdata(qq)-1)/(n-1),0,1)
    pr=ordered(sc,ref); assert pr['rowid'].to_list()==ref and 0<=sc.min() and sc.max()<=1
    pn=f'probe_v21list_{nm}_p33_w80'; pl.DataFrame({'rowid':ref,'predicted_probability':pr['p'].to_numpy()}).write_csv(f'{O}/{pn}.csv',float_precision=9); print(pn,zipit(f'{O}/{pn}.csv'))
