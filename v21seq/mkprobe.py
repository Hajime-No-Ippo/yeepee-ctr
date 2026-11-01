"""mkprobe.py LABEL [A|B ...]  -> out/ probes from C27 ens (ctl6 + v21seq_big seeds that exist). blend = .4*z + .6*M, M = mean logit over all ctl6+big seeds. LABEL tags file names (e.g. s3)."""
import os,sys,glob,hashlib,zipfile,numpy as np,polars as pl
from scipy.special import expit
from scipy.stats import rankdata, spearmanr
H='/workspace/huawei'; os.chdir(f'{H}/v21seq'); label=sys.argv[1]; vars_=sys.argv[2:] or ['A']; O='out'; W_=0.6
ref=pl.read_csv(f'{H}/v5/submission_v5.csv',columns=['rowid'])['rowid'].to_list(); q=pl.read_parquet(f'{H}/teammate/work33/p33_recovered_ranks.parquet').select(['rowid','p33_rank'])
rid=pl.read_parquet(f'{H}/v12/feat_v3/eval.parquet',columns=['rowid'])['rowid'].to_list(); ev=pl.read_csv(f'{H}/data/evaluation.csv',columns=['rowid'])['rowid'].to_list()
assert set(ref)==set(ev)==set(rid)==set(q['rowid'].to_list()) and len(ref)==529390
MG=pl.read_parquet(f'{H}/erbrown/merged.parquet',columns=['rowid']); mrow=MG['rowid'].to_list(); EN=np.load(f'{H}/erbrown/w10an/eb_new_rank.npy'); assert len(mrow)==len(EN)==529390 and set(mrow)==set(ev)
sums=[]
def zipit(csv):
    zp=csv+'.zip'; zf=zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED); zf.write(csv,arcname=os.path.basename(csv)); zf.close(); os.remove(csv)
    sh=hashlib.sha256(open(zp,'rb').read()).hexdigest(); sums.append(f'{sh}  {os.path.basename(zp)}'); return sh
def ordered(vals,order): return pl.DataFrame({'rowid':rid,'p':vals}).join(pl.DataFrame({'rowid':order}),on='rowid',how='right',maintain_order='right')['p'].to_numpy()
def tologit(lp): lp=np.minimum(lp,-1e-6); return lp-np.log1p(-np.exp(lp))
nr=lambda x:(rankdata(x)-1)/(len(x)-1)
for v in vars_:
    nm='B_TRANSDUCTIVE' if v=='B' else 'A_trainonly'
    L_=lambda f,col:np.load(f'{H}/v19/ens/{f}.npy')[:,col]
    b=L_(f'nm{v}s_c27_s0',0); d=L_(f'nm{v}s_s_c27_s0',1); ml=np.mean([L_(f'nm{v}smlp_c27_s{k}',3) for k in (0,1,2)],0); z=(b+d+2*ml)/4
    fs=sorted(glob.glob(f'ens/ctl6_{v}_c27_s*.npy'))+sorted(glob.glob(f'ens/v21seq_big_{v}_c27_s*.npy')); assert fs
    M=np.mean([tologit(np.load(f)[:,-1]) for f in fs],0); assert len(M)==len(z)
    Lg=tologit(np.mean([np.load(f'{H}/v21diag/ens/v21list{v}_c27_s{k}.npy')[:,3] for k in (0,1,2)],0))
    print(v,len(fs),'models; spearman M vs v21list',round(spearmanr(M[::5],Lg[::5])[0],4),'M vs z',round(spearmanr(M[::5],z[::5])[0],4),flush=True)
    z2=(1-W_)*z+W_*M; p=ordered(expit(z2),ev); assert np.isfinite(p).all() and 0<=p.min() and p.max()<=1
    pl_=ordered(expit((1-W_)*z+W_*Lg),ev)   # v21list reference probe (same sample, same rows)
    name=f'submission_v21seq_{nm}_{label}'; pl.DataFrame({'rowid':ev,'predicted_probability':p}).write_csv(f'{O}/{name}.csv'); print(name,zipit(f'{O}/{name}.csv'),'spearman vs v21list',round(spearmanr(p,pl_)[0],5),flush=True)
    qq=pl.DataFrame({'rowid':ev}).join(q,on='rowid',how='left',maintain_order='left')['p33_rank'].to_numpy(); n=len(p)
    sc=np.clip(0.8*(rankdata(p)-1)/(n-1)+0.2*(rankdata(qq)-1)/(n-1),0,1)
    pr=pl.DataFrame({'rowid':ev,'p':sc}).join(pl.DataFrame({'rowid':ref}),on='rowid',how='right',maintain_order='right')['p'].to_numpy(); assert 0<=pr.min() and pr.max()<=1
    pn=f'probe_v21seq_{nm}_p33_w80_{label}'; pl.DataFrame({'rowid':ref,'predicted_probability':pr}).write_csv(f'{O}/{pn}.csv',float_precision=9); print(pn,zipit(f'{O}/{pn}.csv'),flush=True)
    # EB-new mixes (rows in merged.parquet order, rank-scaled like erbrown probes.py)
    pm=pl.DataFrame({'rowid':ev,'p':p}).join(pl.DataFrame({'rowid':mrow}),on='rowid',how='right',maintain_order='right')['p'].to_numpy(); Rk=nr(pm)
    for wE,tag in ((0.9,'w90_10'),(0.85,'w85_15')):
        sc2=wE*EN+(1-wE)*Rk; assert sc2.min()>=0 and sc2.max()<=1 and np.isfinite(sc2).all()
        name=f'probe_ebnew_v21seq_{v}_{tag}_{label}'; pl.DataFrame({'rowid':mrow,'predicted_probability':sc2}).write_csv(f'{O}/{name}.csv',float_precision=12); print(name,zipit(f'{O}/{name}.csv'),'spearman vs EBnew',round(spearmanr(sc2,EN)[0],6),flush=True)
open(f'{O}/SHA256SUMS_{label}.txt','w').write('\n'.join(sums)+'\n')
