import sys,glob,json,numpy as np
from sklearn.metrics import roc_auc_score as AUC
from scipy.stats import rankdata
cut=int(sys.argv[1]); H='/workspace/huawei'; z=np.load(f'{H}/cb_blend/oof_c{cut}.npz',allow_pickle=True); y=z['y']
base=['raw_M','raw_big','raw_A'] if cut==26 else ['raw_z','raw_M','raw_big','raw_D']
def zs(a): a=a.astype(np.float64); return (a-a.mean())/a.std()   # equal logit mean on standardised raw (same as screen.py convention? use plain raw mean too)
B=np.mean([z[k].astype(np.float64) for k in base],0); b0=AUC(y,B); print('cut',cut,'base',base,'AUC',round(b0,5))
out={}
for f in sorted(glob.glob(f'{H}/research/blending_v1_scripts/out/*_c{cut}.npy')):
    n=f.split('/')[-1][:-8]; s=np.load(f).astype(np.float64)
    r=np.corrcoef(rankdata(s),rankdata(B))[0,1]
    # equal member (1/(n+1) share) with new member scaled to the base-mean's std
    sc=s*B.std()/s.std()
    eq=np.mean([z[k].astype(np.float64) for k in base]+[sc+B.mean()-sc.mean()],0)
    d={'single':round(AUC(y,s),5),'rho':round(r,3),'eq_delta':round(AUC(y,eq)-b0,5)}
    for w in (0.1,0.2,0.3): d[f'w{w}']=round(AUC(y,(1-w)*B+w*(sc+B.mean()-sc.mean()))-b0,5)
    out[n]=d; print(n,d)
json.dump(out,open(f'{H}/research/blending_v1_scripts/out/nvblend_c{cut}.json','w'),indent=1)
