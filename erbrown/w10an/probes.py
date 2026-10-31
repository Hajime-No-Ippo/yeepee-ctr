import numpy as np, polars as pl, zipfile, io, hashlib, os
from scipy.stats import rankdata, spearmanr
D=pl.read_parquet('../merged.parquet'); n=D.height; ev=D['event_id'].to_numpy()
z=zipfile.ZipFile('submission_blend_newbest_v21_w10_20261031.csv.zip'); Wd=pl.read_csv(io.BytesIO(z.read(z.namelist()[0])))
assert Wd['rowid'].to_list()==D['rowid'].to_list()
W=Wd['predicted_probability'].to_numpy()
nr=lambda x:(rankdata(x)-1)/(n-1)
O=nr(D['ours'].to_numpy()); P=nr(D['p33_rank'].to_numpy()); EN=np.load('eb_new_rank.npy')
print('control 0.9EN+0.1O max abs err vs w10',np.abs(0.9*EN+0.1*O-W).max(),'spearman EN vs rank-scaled',spearmanr(EN,nr(EN))[0])
codes=np.unique(ev,return_inverse=True)[1]
def smean(x): s=np.bincount(codes,x)/np.bincount(codes); return s[codes]
def zipit(csv):
    zp=csv+'.zip'; zz=zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED); zz.write(csv,arcname=os.path.basename(csv)); zz.close(); os.remove(csv)
    return hashlib.sha256(open(zp,'rb').read()).hexdigest()
out=[]
def mk(name,sc,rescale=False):
    if rescale: sc=nr(sc)
    assert sc.min()>=0 and sc.max()<=1 and np.isfinite(sc).all() and len(sc)==n
    csv=f'out/{name}.csv'; pl.DataFrame({'rowid':D['rowid'],'predicted_probability':sc}).write_csv(csv,float_precision=12)
    sh=zipit(csv); out.append(f'{sh}  {name}.csv.zip')
    print(name,sh,'spearman vs w10',round(spearmanr(sc,W)[0],6),'vs EBnew',round(spearmanr(sc,EN)[0],6),'vs ours',round(spearmanr(sc,O)[0],5),'rows',len(sc))
mk('control_w10_reproduction_EBnew90_ours10',0.9*EN+0.1*O)
mk('probe_ebnew_v21list_B_w95_05',0.95*EN+0.05*O)
mk('probe_ebnew_v21list_B_w80_20',0.80*EN+0.20*O)
mk('probe_ebnew_v21list_p33_B_w85_10_05',0.85*EN+0.10*O+0.05*P)
# slate-mean from EB-new only; within-slate demeaned from blend
blend=0.9*EN+0.1*O
mk('probe_ebnew_v21list_B_w90_10_withinslate_only',blend-smean(blend)+smean(EN),rescale=True)
open('out/SHA256SUMS.txt','w').write('\n'.join(out)+'\n')
