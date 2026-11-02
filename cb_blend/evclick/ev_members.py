import sys; sys.path.insert(0,'.')
import numpy as np, polars as pl, pickle, json
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
B='/workspace/huawei/cb_blend/'
nr=lambda x: rankdata(x)/len(x)
lg=lambda p: np.log(p/(1-p))
WG=[0,.02,.04,.06,.08,.10,.125,.15,.2,.3,.4]
CUTS=(23,25,26)
D={}
for c in CUTS:
    o=np.load(B+f'oof_c{c}.npz',allow_pickle=True); y=o['y']; rid=o['rowid']
    rows=pl.read_parquet(f'rows_val{c}.parquet'); ep=np.load(f'evpred_c{c}.npz')
    pe=pl.DataFrame(dict(eid=ep['eid'],p=ep['p'])); r=rows.join(pe,on='eid',how='left',maintain_order='left')
    mp=dict(zip(r['rowid'].to_numpy(),r['p'].to_numpy())); P=np.array([mp[i] for i in rid]); assert not np.isnan(P).any()
    R={m:nr(o[f'raw_{m}'].astype(np.float64)) for m in ('z','M','big','D') if f'raw_{m}' in o.files}
    g=nr(np.load(B+f'graph0312/graph_c{c}.npy').astype(np.float64))
    ms=('M','big') if c==26 else ('z','M','big','D')
    base=.87*np.mean([R[m] for m in ms],0)+.13*g
    M=o['raw_M'].astype(np.float64)
    D[c]=dict(y=y,P=P,R=R,g=g,base=base,M=M)
    print(c,'rows',len(y),'base AUC',round(roc_auc_score(y,base),5),'M AUC',round(roc_auc_score(y,R['M']),5))
def forms(c):
    d=D[c]; P=d['P']; out={}
    out['A_bcast']=nr(P)
    out['B_prod_M']=nr(nr(P)*d['R']['M']); out['B_prod_eq4']=nr(nr(P)*d['base'])
    out['C_logit_a1']=nr(d['M']+lg(P)); out['C_logit_a.5']=nr(d['M']+.5*lg(P)); out['C_logit_a2']=nr(d['M']+2*lg(P))
    # within-event-normalised item score x event prob: item score = M minus event mean? needs eid; do via polars below
    return out
FM={c:forms(c) for c in CUTS}
rows=[]; res={}
for k in FM[23]:
    r={}
    for c in CUTS:
        d=D[c]; f=FM[c][k]; y=d['y']
        r[f'auc{c}']=roc_auc_score(y,f)
        r[f'rhoM{c}']=np.corrcoef(f,d['R']['M'])[0,1]; r[f'rhoG{c}']=np.corrcoef(f,d['g'])[0,1]; r[f'rhoB{c}']=np.corrcoef(f,d['base'])[0,1]
        b=roc_auc_score(y,d['base']); gains=[roc_auc_score(y,(1-w)*d['base']+w*f)-b for w in WG]; r[f'gains{c}']=gains
        r[f'oracle{c}']=max(gains); r[f'wbest{c}']=WG[int(np.argmax(gains))]
    for a,b_ in ((23,25),(25,23),(23,26),(25,26)):
        w=WG[int(np.argmax(r[f'gains{a}']))]; r[f'ho_fit{a}_on{b_}']=r[f'gains{b_}'][WG.index(w)]; r[f'w_fit{a}']=w
    res[k]=r
    print(k,'AUC',[round(r[f'auc{c}'],4) for c in CUTS],'rhoB',[round(r[f'rhoB{c}'],3) for c in CUTS],'rhoM23 %.3f'%r['rhoM23'],'rhoG23 %.3f'%r['rhoG23'],
      '| oracle gain',[f"{r[f'oracle{c}']*1e4:+.2f}" for c in CUTS],'| HO 23->25 %+.2f 25->23 %+.2f 23->26 %+.2f 25->26 %+.2f (e-4)'%tuple(r[f'ho_fit{a}_on{b_}']*1e4 for a,b_ in ((23,25),(25,23),(23,26),(25,26))),'w',r['w_fit23'],r['w_fit25'])
pickle.dump(res,open('members_res.pkl','wb'))
