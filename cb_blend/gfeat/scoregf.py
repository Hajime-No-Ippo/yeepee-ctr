import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='4'
import sys, json, numpy as np, polars as pl
sys.path.insert(0,'/workspace/huawei/cb_blend'); sys.dont_write_bytecode=True
from blend_oof import rk, AUC
from scipy.stats import spearmanr
H='/workspace/huawei'; B=f'{H}/cb_blend'; G0=f'{B}/gfeat'; V=f'{H}/v21seq'
R={}; D={}
for c in (23,25,26):
    o=np.load(f'{B}/oof_c{c}.npz',allow_pickle=True)
    vr=pl.read_parquet(f'{H}/v12/feat_v3/val{c}.parquet',columns=['rowid','clicked','u'])
    tab=np.load(f'{H}/v21graph/out/tab_val{c}.npz',allow_pickle=True)
    f=np.load(f'{G0}/gfeat_c{c}.npy').astype(np.float64); fc=np.load(f'{G0}/gfeat_control_c{c}.npy').astype(np.float64)
    g=np.load(f'{B}/graph0312/graph_c{c}.npy').astype(np.float64)
    rid=vr['rowid'].cast(pl.Utf8).to_numpy().astype(str)
    ok=dict(rowid_val_tab=bool((rid==tab['rowid'].astype(str)).all()),rowid_val_oof=bool((o['rowid'].astype(str)==rid).all()),y=bool((o['y']==vr['clicked'].to_numpy()).all()))
    assert len(f)==len(fc)==len(g)==len(o['y'])==len(vr) and np.isfinite(f).all() and np.isfinite(fc).all() and all(ok.values()),('MISALIGNED',c,len(f),ok)
    R[f'align_c{c}']=dict(n=len(f),**ok)
    D[c]=dict(y=o['y'],u=vr['u'].to_numpy(),M=rk(o['raw_M']),big=rk(o['raw_big']),G=rk(g),F=rk(f),Fc=rk(fc))
    if c<26: D[c]['z']=rk(o['raw_z']); D[c]['D']=rk(o['raw_D'])
    d=D[c]; R[f'standalone_c{c}']={k:AUC(d['y'],d[k]) for k in ('F','Fc','G','M','big')}
    R[f'rho_c{c}']={'F~M':float(spearmanr(d['F'][::3],d['M'][::3])[0]),'F~G':float(spearmanr(d['F'][::3],d['G'][::3])[0]),'Fc~M':float(spearmanr(d['Fc'][::3],d['M'][::3])[0]),'G~M':float(spearmanr(d['G'][::3],d['M'][::3])[0])}
print(json.dumps(R,indent=1))
def eq4(c): d=D[c]; return (d['z']+d['M']+d['big']+d['D'])/4
def mb(c): d=D[c]; return (d['M']+d['big'])/2
WG=np.round(np.arange(0,.3001,.005),3); WF=np.round(np.arange(0,.4001,.02),3)
def grid(c,base,fk):
    d=D[c]; b=base; y=d['y']; A=np.zeros((len(WG),len(WF)))
    # w_f=0 column first; then joint: score=(1-wg-wf)*b + wg*G + wf*F ; wg coarse step .01 for speed
    for i,wg in enumerate(WG):
        if i%2: A[i]=np.nan; continue
        for j,wf in enumerate(WF):
            if wg+wf>=.7: A[i,j]=np.nan; continue
            A[i,j]=AUC(y,(1-wg-wf)*b+wg*d['G']+wf*d[fk])
    return A
res={}
for fk in ('F','Fc'):
    for c in (23,25,26):
        base=eq4(c) if c<26 else mb(c)
        res[(fk,c)]=grid(c,base,fk)
def best(A,mask_wf0=False):
    A2=A.copy()
    if mask_wf0: A2[:,1:]=np.nan
    i,j=np.unravel_index(np.nanargmax(A2),A2.shape); return float(WG[i]),float(WF[j]),float(A[i,j])
def at(A,wg,wf):
    i=int(np.argmin(abs(WG-wg))); i-=i%2 if WG[i]!=wg and False else 0
    # snap wg to even grid (step .01)
    wg=round(round(wg/.01)*.01,3); i=list(WG).index(wg); j=list(WF).index(round(round(wf/.02)*.02,3)); return float(A[i,j])
O={}
for fk in ('F','Fc'):
    r={}
    # baseline eq4+graph: wg fit on other fold (wf=0)
    wgs={c:best(res[(fk,c)],True) for c in (23,25,26)}
    jts={c:best(res[(fk,c)]) for c in (23,25,26)}
    r['eq4+graph wg-only oracle']={c:wgs[c] for c in (23,25,26)}
    r['joint oracle (wg,wf,auc)']={c:jts[c] for c in (23,25,26)}
    for tgt,src in ((25,23),(23,25),(26,23),(26,25)):
        wg0=wgs[src][0]; base_auc=at(res[(fk,tgt)],wg0,0)            # eq4+graph at held-out wg
        wgj,wfj,_=jts[src]; a_j=at(res[(fk,tgt)],wgj,wfj)           # joint (wg,wf) held-out
        # wf only, wg fixed at src wg0 (re-fit wf on src with wg fixed)
        A=res[(fk,src)]; i=list(WG).index(round(round(wg0/.01)*.01,3)); wf_fix=float(WF[np.nanargmax(A[i])]); a_f=at(res[(fk,tgt)],wg0,wf_fix)
        r[f'tgt C{tgt} <- fit C{src}']=dict(base_eq4_graph=(wg0,base_auc),joint=(wgj,wfj,a_j,(a_j-base_auc)*1e4),wf_only_wg_fixed=(wg0,wf_fix,a_f,(a_f-base_auc)*1e4))
    # also ref gain from graph (vs eq4 alone) for context
    r['eq4 alone C23/C25']=[float(res[(fk,c)][0,0]) for c in (23,25)]; r['mb alone C26']=float(res[(fk,26)][0,0])
    O[fk]=r
print(json.dumps(O,indent=1,default=float))
# --- replacement of M/big by gfeat (graph weight refit, wf=0 at both): base = z,D,M',big' ; F takes share
def repl(c,rM,rb,fk='F'):
    d=D[c]; return (d['z']+(1-rM)*d['M']+(1-rb)*d['big']+d['D']+(rM+rb)*d[fk])/4
RP={}
for name,(rM,rb) in {'F replaces M 50%':(.5,0),'F replaces M 100%':(1,0),'F replaces big 50%':(0,.5),'F replaces big 100%':(0,1),'F replaces M+big 50%':(.5,.5),'F replaces M+big 100%':(1,1)}.items():
    cur={}
    for c in (23,25):
        cur[c]=[AUC(D[c]['y'],(1-w)*repl(c,rM,rb)+w*D[c]['G']) for w in WG[::2]]
    wsel={c:float(WG[::2][int(np.argmax(cur[c]))]) for c in cur}
    base={c:[AUC(D[c]['y'],(1-w)*eq4(c)+w*D[c]['G']) for w in WG[::2]] for c in (23,25)} if not RP else None
    if base: RP['_base']=base
    ix=lambda w: list(WG[::2]).index(round(w,3))
    out={}
    for tgt,src in ((25,23),(23,25)):
        w=wsel[src]; b_w=float(WG[::2][int(np.argmax(RP['_base'][src]))]); out[f'C{tgt}<-{src}']=dict(w=w,auc=cur[tgt][ix(w)],base_w=b_w,base_auc=RP['_base'][tgt][ix(b_w)],gain_e4=(cur[tgt][ix(w)]-RP['_base'][tgt][ix(b_w)])*1e4)
    RP[name]=out
RP.pop('_base'); print(json.dumps(RP,indent=1))
# --- C26 M/big 50-50 + graph + gfeat details
# --- user-hash group uncertainty of the joint held-out candidate vs baseline
def grp(c,s_new,s_base,G=8):
    gidx=pl.Series(D[c]['u']).hash(seed=3).to_numpy()%G; y=D[c]['y']
    d=[AUC(y[gidx==i],s_new[gidx==i])-AUC(y[gidx==i],s_base[gidx==i]) for i in range(G)]; return float(np.mean(d)),float(np.std(d,ddof=1))
U={}
for tgt,src in ((25,23),(23,25)):
    wg0=O['F']['eq4+graph wg-only oracle'][src][0]; wgj,wfj,_=O['F']['joint oracle (wg,wf,auc)'][src]; d=D[tgt]
    U[f'C{tgt}']=grp(tgt,(1-wgj-wfj)*eq4(tgt)+wgj*d['G']+wfj*d['F'],(1-wg0)*eq4(tgt)+wg0*d['G'])
print('groupdelta mean,sd',U)
json.dump(dict(R=R,O=O,RP=RP,U=U,WG=list(WG),WF=list(WF)),open(f'{G0}/results.json','w'),indent=1,default=float)
np.savez(f'{G0}/grids.npz',**{f'{fk}_{c}':res[(fk,c)] for (fk,c) in res})
