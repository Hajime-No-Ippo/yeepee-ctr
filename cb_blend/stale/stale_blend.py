import numpy as np, pickle, sys
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
sys.path.insert(0,'../evclick')
from evlib import *
B='/workspace/huawei/cb_blend/'
nr=lambda x: rankdata(x)/len(x); WG=[0,.02,.04,.06,.08,.10,.125,.15,.2,.3]
CUTS=[c for c in (23,25,26) if os.path.exists(f'pred_t_c{c}.npy')]
D={}
for c in CUTS:
    o=np.load(B+f'oof_c{c}.npz',allow_pickle=True); y=o['y']; R={m:nr(o[f'raw_{m}'].astype(np.float64)) for m in ('z','M','big','D') if f'raw_{m}' in o.files}
    g=nr(np.load(B+f'graph0312/graph_c{c}.npy').astype(np.float64)); ms=('M','big') if c==26 else ('z','M','big','D')
    base=.87*np.mean([R[m] for m in ms],0)+.13*g
    d,_=cols(f'val{c}',[]); mp=dict(zip(d['rowid'].to_numpy(),d['tm_stale'].to_numpy())); t=np.array([mp[i] for i in o['rowid']])
    D[c]=dict(y=y,base=base,g=g,M=R['M'],t=t)
res={}
for k in ('base80','w','wt','t'):
    for c in CUTS:
        d=D[c]; y=d['y']; f=nr(np.load(f'pred_{k}_c{c}.npy').astype(np.float64)); st=d['t']>86400
        b=roc_auc_score(y,d['base']); gains=[roc_auc_score(y,(1-w)*d['base']+w*f)-b for w in WG]
        r=dict(auc=roc_auc_score(y,f),rhoM=np.corrcoef(f,d['M'])[0,1],rhoG=np.corrcoef(f,d['g'])[0,1],rhoB=np.corrcoef(f,d['base'])[0,1],gains=gains)
        if st.any():
            r['stale_auc']=roc_auc_score(y[st],f[st]); r['stale_base']=roc_auc_score(y[st],d['base'][st])
            r['stale_gain_at_w']=[roc_auc_score(y[st],(1-w)*d['base'][st]+w*f[st])-r['stale_base'] for w in WG]
        res[(k,c)]=r
    out=[]
    for c in CUTS:
        r=res[(k,c)]; out.append(f"C{c} AUC {r['auc']:.4f} rhoB {r['rhoB']:.3f} rhoM {r['rhoM']:.3f} orc {max(r['gains'])*1e4:+.2f}@{WG[int(np.argmax(r['gains']))]}"+(f" | stale-subset AUC {r['stale_auc']:.4f} (eq4 {r['stale_base']:.4f}) stale-gain best {max(r['stale_gain_at_w'])*1e4:+.1f}e-4" if 'stale_auc' in r else ''))
    for a,b2 in ((23,25),(25,23),(23,26),(25,26)):
        if a in CUTS and b2 in CUTS:
            w=WG[int(np.argmax(res[(k,a)]['gains']))]; out.append(f'HO fit C{a}->C{b2}: w={w} gain {res[(k,b2)]["gains"][WG.index(w)]*1e4:+.2f}e-4')
    print(k,'\n  '+'\n  '.join(out),flush=True)
pickle.dump(res,open('stale_res.pkl','wb'))
