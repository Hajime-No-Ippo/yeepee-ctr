exec(open('/workspace/huawei/cb_blend/ensig/ensig.py').read().split('# ---- S1 rebuilt')[0])   # imports + helpers only (no eval loads needed)
import pickle, time
exec("def ld_seeds_loc(pat,col=-1):\n    fs=sorted(glob.glob(H+pat)); assert fs,pat; P=[]\n    for f in fs:\n        a=np.load(f); a=a[:,col] if a.ndim>1 else a; a=a.astype(np.float64); P.append(tologit(a) if a.max()<=0 else a)\n    return np.mean(P,0),len(fs)")
src=open('/workspace/huawei/cb_blend/ensig/ensig.py').read()
# pull REG local specs without executing eval loaders: exec REG block only
reg_src=src[src.index('def L(p,col=-1)'):]
EN=E0=S1=A_=D_=V8=V5=Pq=Oo=None
def ld_seeds(*a,**k): raise RuntimeError
def ld_rowid(*a,**k): raise RuntimeError
exec(reg_src)
LOC={k:r['loc'] for k,r in REG.items()}
# local-only members (NO eval prediction exists anywhere): cat (v21cat numte), ref484 (v20 LightGBM), gfeat (v21gfeat LGB+graph feats)
REG['cat_numte(v21cat)']=dict(fam='LGB + target-encoded cats (v21cat)'); LOC['cat_numte(v21cat)']={c:L(f'v21cat/out/cat_numte_c{c}.npy') for c in (23,25,26)}
REG['ref484(v20)']=dict(fam='LGB 484 feats (v20)'); LOC['ref484(v20)']={c:L(f'v20/ens/ref484_c{c}_s*.npy') for c in (23,25)}
REG['gf_num_s3(v21gfeat)']=dict(fam='LGB + graph feats (v21gfeat)'); LOC['gf_num_s3(v21gfeat)']={c:L(f'v21gfeat/ens/gf_num_s3_c{c}_s0.npy') for c in (23,25,26)}
REG['gf_numg_s3(v21gfeat)']=dict(fam='LGB + graph feats+g (v21gfeat)'); LOC['gf_numg_s3(v21gfeat)']={c:L(f'v21gfeat/ens/gf_numg_s3_c{c}_s0.npy') for c in (23,25,26)}

t0=time.time(); CUTS=(23,25,26)
oof={c:np.load(H+f'cb_blend/oof_c{c}.npz',allow_pickle=True) for c in CUTS}
print({c:[k for k in oof[c].files if k.startswith('raw_')] for c in CUTS})
Y={c:oof[c]['y'] for c in CUTS}
def getloc(k,c):
    spec=LOC[k].get(c)
    if spec is None: return None
    if spec[0]=='GRAPH': return np.load(H+f'cb_blend/graph0312/graph_c{c}.npy').astype(np.float64)
    v,_=ld_seeds_loc(spec[0],spec[1]); assert len(v)==len(Y[c]),(k,c,len(v)); return v
RK={c:{m:rk for m,rk in []} for c in CUTS}
base={}; RKm={c:{} for c in CUTS}
def rk(x): return nr(x)
for c in CUTS:
    for m in ('z','M','big','D','graph0312'):
        if f'raw_{m}' in oof[c].files: RKm[c][m]=rk(oof[c][f'raw_{m}'].astype(np.float64))
for c in CUTS: RKm[c]['graph0312']=rk(np.load(H+f'cb_blend/graph0312/graph_c{c}.npy').astype(np.float64))
W=.13
for c in (23,25): base[c]=(1-W)*np.mean([RKm[c][m] for m in ('z','M','big','D')],0)+W*RKm[c]['graph0312']
base[26]=(1-W)*np.mean([RKm[26][m] for m in ('M','big')],0)+W*RKm[26]['graph0312']
auc=lambda c,s:roc_auc_score(Y[c],s)
B0={c:auc(c,base[c]) for c in CUTS}; print('base eq4+graph(.13):',B0)
WG=[0,.02,.04,.06,.08,.10,.125,.15,.2,.3]
R={}
for k in REG:
    r={'aucs':{},'rhoM':{},'curve':{}}
    for c in CUTS:
        v=getloc(k,c)
        if v is None: continue
        x=rk(v); r['aucs'][c]=auc(c,v)
        r['rhoM'][c]=np.corrcoef(x,RKm[c]['M'])[0,1]
        r['curve'][c]=np.array([auc(c,(1-w)*base[c]+w*x)-B0[c] for w in WG])
    R[k]=r; print(k,{c:round(a,4) for c,a in r['aucs'].items()},'rhoM',{c:round(a,3) for c,a in r['rhoM'].items()},'best23/25/26',[ (WG[int(np.argmax(r['curve'][c]))],round(float(r['curve'][c].max())*1e4,2)) for c in CUTS if c in r['curve']],round(time.time()-t0),flush=True)
pickle.dump(dict(R=R,B0=B0,WG=WG),open(OUT+'local_gain.pkl','wb'))
