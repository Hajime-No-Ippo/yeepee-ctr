import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ['k' if False else k]='4'
import sys, json, numpy as np, polars as pl
sys.path.insert(0,'/workspace/huawei/cb_blend'); sys.dont_write_bytecode=True
from blend_oof import rk, AUC
from scipy.stats import spearmanr
H='/workspace/huawei'; G0=f'{H}/cb_blend/graph0312'; V=f'{H}/v21graph/out'
sys.path.insert(0,f'{H}/cb_blend/u1u2'); from graph_member import graph_member
R={}; D={}
for c in (23,25,26):
    o=np.load(f'{H}/cb_blend/oof_c{c}.npz',allow_pickle=True)
    vr=pl.read_parquet(f'{H}/v12/feat_v3/val{c}.parquet',columns=['rowid','clicked'])
    tab=np.load(f'{V}/tab_val{c}.npz',allow_pickle=True)
    g=np.load(f'{G0}/graph_c{c}.npy').astype(np.float64)
    ok1=(vr['rowid'].cast(pl.Utf8).to_numpy().astype(str)==tab['rowid'].astype(str)).all()
    ok2=(o['rowid'].astype(str)==vr['rowid'].cast(pl.Utf8).to_numpy().astype(str)).all()
    ok3=(o['y']==vr['clicked'].to_numpy()).all() and (tab['y']==o['y']).all()
    assert len(g)==len(o['y'])==len(vr)==len(tab['rowid']) and np.isfinite(g).all(), (c,len(g),len(o['y']))
    assert ok1 and ok2 and ok3, ('MISALIGNED',c,ok1,ok2,ok3)
    R[f'align_c{c}']=dict(n=len(g),rowid_val_vs_tab=bool(ok1),rowid_val_vs_oof=bool(ok2),y_ok=bool(ok3))
    # also check no per-slate demeaning: slate means of raw score vary
    sm=g.reshape(-1,10).mean(1); R[f'slate_mean_sd_c{c}']=float(sm.std()); R[f'global_sd_c{c}']=float(g.std())
    d={'y':o['y'],'M':rk(o['raw_M']),'big':rk(o['raw_big']),'G':rk(g),'GU1':rk(o['raw_graphU1']),'GU1raw':rk(graph_member(c,demean=False))}
    if c<26: d['z']=rk(o['raw_z']); d['D']=rk(o['raw_D'])
    D[c]=d
# eval side: length/rowid only
E=np.load(f'{G0}/graph_c27.npy'); te=np.load(f'{V}/tab_eval.npz',allow_pickle=True)
ev=np.load(f'{V}/graph_eval_v21seqorder.npy'); er=np.load(f'{V}/graph_eval_rowid.npy',allow_pickle=True)
R['eval']=dict(len_c27=len(E),len_tab_eval=len(te['rowid']),c27_eq_v21seqorder=bool(np.allclose(E,ev)),rowid_eq=bool((er.astype(str)==te['rowid'].astype(str)).all()),finite=bool(np.isfinite(E).all()))
for c in D:
    d=D[c]; R[f'standalone_c{c}']={k:AUC(d['y'],d[k]) for k in ('G','GU1','GU1raw','M','big')}
    R[f'rho_M_c{c}']={k:float(spearmanr(d[k][::3],d['M'][::3])[0]) for k in ('G','GU1','GU1raw')}
    R[f'rho_G_vs_GU1_c{c}']=float(spearmanr(d['G'][::3],d['GU1'][::3])[0])
    R[f'rho_G_vs_GU1raw_c{c}']=float(spearmanr(d['G'][::3],d['GU1raw'][::3])[0])
WS=np.round(np.arange(0,0.3001,0.005),3)
def base(c,kind):
    d=D[c]
    return (d['z']+d['M']+d['big']+d['D'])/4 if kind=='eq4' else (d['M']+d['big'])/2
def curve(c,kind,gk):
    d=D[c]; b=base(c,kind); return np.array([AUC(d['y'],(1-w)*b+w*d[gk]) for w in WS])
C={}
for gk in ('G','GU1','GU1raw'):
    for c in (23,25): C[(c,'eq4',gk)]=curve(c,'eq4',gk)
    C[(26,'mb',gk)]=curve(26,'mb',gk)
    for c in (23,25): C[(c,'mb',gk)]=curve(c,'mb',gk)
out={}
for gk in ('G','GU1','GU1raw'):
    r={}
    for c in (23,25): r[f'eq4 C{c} base']=float(C[(c,'eq4',gk)][0]); r[f'eq4 C{c} w=.10 (.225x4+.10)']=float(C[(c,'eq4',gk)][list(WS).index(.1)]); r[f'eq4 C{c} oracle w']=float(WS[C[(c,'eq4',gk)].argmax()]); r[f'eq4 C{c} oracle auc']=float(C[(c,'eq4',gk)].max())
    w23=float(WS[C[(23,'eq4',gk)].argmax()]); w25=float(WS[C[(25,'eq4',gk)].argmax()])
    r['heldout: w fit on C23, scored on C25']=(w23,float(C[(25,'eq4',gk)][list(WS).index(w23)]))
    r['heldout: w fit on C25, scored on C23']=(w25,float(C[(23,'eq4',gk)][list(WS).index(w25)]))
    # C26: M/big 50-50 + graph
    b=C[(26,'mb',gk)]; r['mb C26 base']=float(b[0]); r['mb C26 oracle']=(float(WS[b.argmax()]),float(b.max()))
    for lab,w in (('w from C23',w23),('w from C25',w25),('w=.10',.1),('w=.125',.125),('w=.15',.15)): r[f'mb C26 {lab}']=(w,float(b[list(WS).index(w)]))
    for c in (23,25): r[f'mb C{c} base/w.10/oracle']=(float(C[(c,'mb',gk)][0]),float(C[(c,'mb',gk)][list(WS).index(.1)]),float(WS[C[(c,'mb',gk)].argmax()]),float(C[(c,'mb',gk)].max()))
    out[gk]=r
R['results']=out
R['curves']={f'{k[0]}_{k[1]}_{k[2]}':[float(x) for x in v[::4]] for k,v in C.items()}
R['curve_ws']=[float(x) for x in WS[::4]]
json.dump(R,open(f'{G0}/results.json','w'),indent=1,default=float)
print(json.dumps({k:v for k,v in R.items() if k not in('curves','curve_ws')},indent=1,default=float))
