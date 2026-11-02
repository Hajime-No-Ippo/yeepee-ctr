import numpy as np,json,os,sys
from sklearn.metrics import roc_auc_score as AUC
H='out'; E='/workspace/huawei/v20/ens'; res={}
def z(a): return (a-a.mean())/a.std()
for c in (23,25):
    o=np.load(f'/workspace/huawei/cb_blend/oof_c{c}.npz'); y=o['y']
    refs=[np.load(f'{E}/ref484_c{c}_s{s}.npy') for s in range(3)]
    aref=[AUC(y,r) for r in refs]; base=np.mean([z(o[k]) for k in('raw_z','raw_M','raw_big','raw_D')],0)
    d={'ref_s0':aref[0],'ref_mean3':float(np.mean(aref)),'blend4':AUC(y,base),'blend4+ref484s0':AUC(y,np.mean([z(o[k]) for k in('raw_z','raw_M','raw_big','raw_D')]+[z(refs[0])],0))}
    for t in ('tmx','sq4'):
        f=f'{H}/fs_{t}_c{c}_s0.npy'
        if not os.path.exists(f): continue
        s=np.load(f); assert len(s)==len(y)
        d[t]=AUC(y,s); d[t+'_vs_s0']=d[t]-aref[0]; d[t+'_vs_mean3']=d[t]-np.mean(aref)
        d[t+'_corr_ref']=float(np.corrcoef(s,refs[0])[0,1])
        b5=AUC(y,np.mean([z(o[k]) for k in('raw_z','raw_M','raw_big','raw_D')]+[z(s)],0))
        d['blend4+'+t]=b5; d[t+'_blend_delta_vs_ref_member']=b5-d['blend4+ref484s0']; d[t+'_blend_delta_vs_blend4']=b5-d['blend4']
    res[c]=d
json.dump(res,open(f'{H}/fs490_eval.json','w'),indent=1,default=float)
for c,d in res.items(): print(c,{k:round(v,5) for k,v in d.items()})
