"""blend_new.py (2026-11-02): judge 'cat' and 'graph' on top of eq4 (z,M,big,D .25) with weights chosen on ONE fold, scored on the OTHER.
Reads cached oof_c{23,25,26}.npz only (built by blend_oof.py --rebuild). No eval/submission output."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ.setdefault(k,'4')
import json, numpy as np, polars as pl
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score as AUC
OUT='/workspace/huawei/cb_blend'
def rk(x): return (rankdata(x)-1)/(len(x)-1)
D={}
for c in (23,25,26):
    z=np.load(f'{OUT}/oof_c{c}.npz'); D[c]=dict(y=z['y'],raw={k[4:]:z[k].astype(np.float64) for k in z.files if k.startswith('raw_')})
    D[c]['rk']={m:rk(v) for m,v in D[c]['raw'].items()}
    D[c]['u']=pl.read_parquet(f'/workspace/huawei/v12/feat_v3/val{c}.parquet',columns=['u'])['u'].to_numpy()
R={}
def ug(c,a,b,G=8):
    u=D[c]['u']; g=pl.Series(u).hash(seed=3).to_numpy()%G; y=D[c]['y']; d=[AUC(y[g==i],a[g==i])-AUC(y[g==i],b[g==i]) for i in range(G)]; return np.mean(d)*1e5,np.std(d,ddof=1)*1e5
print('## standalone + rho vs M (Spearman, all rows)')
for c in (23,25,26):
    for m in ('cat','graph'):
        if m in D[c]['rk']:
            a=AUC(D[c]['y'],D[c]['raw'][m]); r=spearmanr(D[c]['raw'][m],D[c]['raw']['M'])[0]; print(c,m,f'auc {a:.5f} rho_M {r:.4f}'); R[f'{m}_C{c}']=dict(auc=a,rhoM=r)
def S(c,W): return sum(w*D[c]['rk'][m] for m,w in W.items())
EQ4={'z':.25,'M':.25,'big':.25,'D':.25}; BASE={'z':.4,'M':.6}
eq4={c:S(c,EQ4) for c in (23,25)}; base={c:S(c,BASE) for c in (23,25)}
A=lambda c,s:AUC(D[c]['y'],s)
print('eq4',{c:round(A(c,eq4[c]),5) for c in eq4},'base',{c:round(A(c,base[c]),5) for c in base})
WS=[0,.025,.05,.075,.1,.125,.15,.2,.25,.3]
print('\n## (1-w)*eq4 + w*new (rank), and same on base .4z+.6M')
for m in ('cat','graph'):
    for bn,B in (('eq4',eq4),('base',base)):
        tab={c:{w:A(c,(1-w)*B[c]+w*D[c]['rk'][m]) for w in WS} for c in (23,25)}
        print(m,'on',bn,' | '.join(f'C{c} '+' '.join(f'{w}:{(tab[c][w]-tab[c][0])*1e5:+.1f}' for w in WS) for c in (23,25)))
        for fa,fb in ((25,23),(23,25)):
            w=max(WS,key=lambda w:tab[fa][w]); R[f'{m}_on_{bn}_fitC{fa}']=dict(w=w,held_cut=fb,held_auc=tab[fb][w],delta_vs_eq4_1e5=(tab[fb][w]-A(fb,eq4[fb]))*1e5)
            print(f'   fit C{fa} -> w={w}  held-out C{fb} {tab[fb][w]:.5f} (vs eq4 {(tab[fb][w]-A(fb,eq4[fb]))*1e5:+.1f}e-5, vs base {(tab[fb][w]-A(fb,base[fb]))*1e5:+.1f}e-5)')
# both
print('\n## eq4 + wc*cat + wg*graph grid, fit on one fold -> other')
G2=[(a,b) for a in (0,.05,.1,.15) for b in (0,.05,.1,.15,.2) if a+b<=.3]
tab={c:{(a,b):A(c,(1-a-b)*eq4[c]+a*D[c]['rk']['cat']+b*D[c]['rk']['graph']) for a,b in G2} for c in (23,25)}
for fa,fb in ((25,23),(23,25)):
    k=max(G2,key=lambda k:tab[fa][k]); print(f'fit C{fa} -> (cat,graph)={k} held C{fb} {tab[fb][k]:.5f} ({(tab[fb][k]-A(fb,eq4[fb]))*1e5:+.1f}e-5 vs eq4)'); R[f'both_fitC{fa}']=dict(w=k,held=tab[fb][k])
# C26 cat on M/big
y=D[26]['y']; mb=.5*D[26]['rk']['M']+.5*D[26]['rk']['big']
print('\nC26 M/big50 + w*cat:',' '.join(f'{w}:{(AUC(y,(1-w)*mb+w*D[26]["rk"]["cat"])-AUC(y,mb))*1e5:+.1f}' for w in WS))
json.dump(R,open(f'{OUT}/blend_new.json','w'),indent=1,default=float)
import pickle; pickle.dump(dict(tab=None),open('/dev/null','wb'))
