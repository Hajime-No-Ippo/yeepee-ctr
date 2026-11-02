"""Section-9 evaluator: python fsx_eval.py  -> out/fsx_eval.json. Deltas of each fsx_{tag} single model vs 490 (fs_tmx s0) and vs 484 (ref484 s0 / mean3), and equal-z blend (z,M,big,D + model as 5th member) vs blend4+490."""
import numpy as np,json,os,glob
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score as AUC
H='out'; E='/workspace/huawei/v20/ens'; res={}
def z(a): return (a-a.mean())/a.std()
tags=sorted({os.path.basename(f)[4:].rsplit('_c',1)[0] for f in glob.glob(f'{H}/fsx_*_c2?_s0.npy')})
for c in (23,25):
    o=np.load(f'/workspace/huawei/cb_blend/oof_c{c}.npz'); y=o['y']; four=[z(o[k]) for k in('raw_z','raw_M','raw_big','raw_D')]
    refs=[np.load(f'{E}/ref484_c{c}_s{s}.npy') for s in range(3)]; aref=[AUC(y,r) for r in refs]
    s490=np.load(f'{H}/fs_tmx_c{c}_s0.npy'); a490=AUC(y,s490); b4=AUC(y,np.mean(four,0)); b490=AUC(y,np.mean(four+[z(s490)],0))
    rb4=rankdata(np.mean(four,0))
    d={'ref484_s0':aref[0],'ref484_mean3':float(np.mean(aref)),'m490':a490,'blend4':b4,'blend4+490':b490}
    for t in tags:
        for sd in (0,1):
            f=f'{H}/fsx_{t}_c{c}_s{sd}.npy'
            if not os.path.exists(f): continue
            s=np.load(f); assert len(s)==len(y); k=t if sd==0 else t+'_s1'
            a=AUC(y,s); b=AUC(y,np.mean(four+[z(s)],0))
            d[k]=dict(auc=a,vs490=a-a490,vs484_s0=a-aref[0],vs484_mean3=a-np.mean(aref),corr490=float(np.corrcoef(s,s490)[0,1]),rho_blend4=float(np.corrcoef(rankdata(s),rb4)[0,1]),blend5=b,blend5_vs_blend4=b-b4,blend5_vs_blend4_490=b-b490)
    res[c]=d
json.dump(res,open(f'{H}/fsx_eval.json','w'),indent=1,default=float)
for c,d in res.items():
    print('C',c,{k:round(v,5) for k,v in d.items() if not isinstance(v,dict)})
    for k,v in d.items():
        if isinstance(v,dict): print('  ',k,{a:round(b,5) for a,b in v.items()})
