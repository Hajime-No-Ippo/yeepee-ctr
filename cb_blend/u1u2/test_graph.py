import sys,glob,json,os; sys.dont_write_bytecode=True; sys.path.insert(0,'/workspace/huawei/cb_blend')
import numpy as np
import blend_oof as B
from blend_oof import AUC, rk
D='/workspace/huawei/cb_blend/u1u2'
data=B.build()
TAG={23:'u1b',25:'u1',26:'u1'}
def seeds(c,tag=None): 
    fs=sorted(glob.glob(f'{D}/feat_c{c}_{tag or TAG[c]}_s*.npz'),key=lambda f:int(f.split('_s')[-1][:-4])); return fs
def zs(x): return (x-x.mean())/x.std()
def dm(s): return (s.reshape(-1,10)-s.reshape(-1,10).mean(1,keepdims=True)).ravel()
def member(c,fs,demean=True,parts=('s_uv','s_h','cos_mean')):
    acc={k:0 for k in parts}
    for f in fs:
        e=np.load(f)
        for k in parts: acc[k]=acc[k]+e[k].astype(np.float64)
    f_=dm if demean else (lambda x:x)
    return sum(zs(f_(acc[k]/len(fs))) for k in parts)
def base_sets(c):
    r=data[c]['rk']; b={'base':.4*r['z']+.6*r['M']} if 'z' in r else {}
    if 'z' in r and 'D' in r and 'big' in r: b['eq4']=.25*(r['z']+r['M']+r['big']+r['D'])
    if 'z' in r and 'D' in r: b['stackD']=.3*r['z']+.3*r['M']+.4*r['D']
    if c==26: b['A50']=.5*(r['M']+r['big'])
    return b
WS=np.round(np.arange(0,0.301,0.025),3)
def gain_curve(c,G,bk):
    y=data[c]['y']; b=base_sets(c)[bk]; g=rk(G); a0=AUC(y,b); return {float(w):AUC(y,(1-w)*b+w*g)-a0 for w in WS},a0
out={}
def P(*a): print(*a,flush=True)
# ---------- (2) reproduction ----------
P('== 2. Reproduce Grok gains (.4z+.6M base, rank blend (1-w)b+wG, dm sum3, all available seeds) ==')
G={}
for c in (23,25,26):
    fs=seeds(c); G[c]=member(c,fs); y=data[c]['y']
    rho=np.corrcoef(rk(G[c]),data[c]['rk']['M'])[0,1]
    P(f'C{c}: n_seeds={len(fs)} standalone AUC {AUC(y,G[c]):.5f} rho_M(pearson on ranks) {rho:.3f}')
    for bk in base_sets(c):
        cur,a0=gain_curve(c,G[c],bk); P(f'   vs {bk} (AUC {a0:.5f}): '+' '.join(f'w{w}:{cur[w]*1e5:+.0f}' for w in (.05,.1,.15,.2,.3)))
# Grok's interim table used n=5 (C25) / n=4 (C23): replicate with first-n seeds
for c,n in ((25,5),(23,4)):
    fs=seeds(c)[:n]; g=member(c,fs); cur,a0=gain_curve(c,g,'base'); P(f'C{c} first {n} seeds (Grok interim cfg): AUC {AUC(data[c]["y"],g):.5f} '+' '.join(f'w{w}:{cur[w]*1e5:+.0f}' for w in (.05,.1,.15,.2)))
# ---------- (3) held-out ----------
P('\n== 3. Held-out w (choose w on one fold by max gain, score on the other) ==')
def heldout(Gd,bk,label):
    res={}
    for fit,te in ((23,25),(25,23)):
        cf,_=gain_curve(fit,Gd[fit],bk); w=max(cf,key=cf.get); ct,a0=gain_curve(te,Gd[te],bk)
        # user-subset sd of the held-out delta
        b=base_sets(te)[bk]; s=(1-w)*b+w*rk(Gd[te]); m,sd=B.user_group_delta(data,te,s,b)
        res[f'fit{fit}->C{te}']=dict(w=w,gain_fit=cf[w],gain_test=ct[w],gain_test_w1=ct[.1],base_auc=a0,sub_sd=sd)
    mean=np.mean([v['gain_test'] for v in res.values()])
    P(f'{label:34s} vs {bk:7s}: '+' | '.join(f'{k}: w={v["w"]:.3f} in-fold {v["gain_fit"]*1e5:+.0f} HELD {v["gain_test"]*1e5:+.0f} (±{v["sub_sd"]*1e5:.0f}) [w.1 {v["gain_test_w1"]*1e5:+.0f}]' for k,v in res.items())+f' | mean held {mean*1e5:+.1f}e-5')
    out[f'{label}|{bk}']=res; return res
for bk in ('base','eq4'):
    heldout(G,bk,'dm sum3 (Grok)')
Graw={c:member(c,seeds(c),demean=False) for c in (23,25)}
for bk in ('base','eq4'):
    heldout(Graw,bk,'RAW sum3 (no slate demean)')
Gs2={c:member(c,seeds(c),parts=('s_uv','s_h')) for c in (23,25)}
heldout(Gs2,'base','dm s_uv+s_h')
Gcm={c:zs(dm(np.mean([np.load(f)['cos_mean'] for f in seeds(c)],0).astype(np.float64))) for c in (23,25)}
heldout(Gcm,'base','dm cos_mean')
# fixed-w references (no tuning): w=.1
P('\n== C26 (only M,big exist; graph u1 k6) ==')
y=data[26]['y']; for_w=np.mean([np.mean([out['dm sum3 (Grok)|base'][k]['w'] for k in out['dm sum3 (Grok)|base']])])
b=base_sets(26)['A50']; a0=AUC(y,b); P(f'M/big 50-50 base {a0:.5f}; M alone {AUC(y,data[26]["rk"]["M"]):.5f}; graph standalone {AUC(y,G[26]):.5f}')
cur,_=gain_curve(26,G[26],'A50'); P('  gain vs M/big: '+' '.join(f'w{w}:{cur[w]*1e5:+.0f}' for w in (.025,.05,.075,.1,.15,.2,.3)))
# ---------- seed sensitivity ----------
P('\n== Seed sensitivity (dm sum3; gain vs .4z+.6M at w=.1 / best-w; standalone AUC) ==')
for c in (23,25,26):
    fs=seeds(c); P(f'C{c} {TAG[c]} single seeds:')
    for f in fs:
        g=member(c,[f]); cur,a0=gain_curve(c,g,'base') if 'z' in data[c]['rk'] else gain_curve(c,g,'A50'); bk=max(cur,key=cur.get)
        P(f'   {os.path.basename(f)[11:-4]:8s} AUC {AUC(data[c]["y"],g):.5f} gain w.1 {cur[.1]*1e5:+.0f} best {cur[bk]*1e5:+.0f}@{bk}')
    for n in sorted({1,2,3,len(fs)}):
        if n>len(fs): continue
        g=member(c,fs[:n]); cur,a0=gain_curve(c,g,'base') if 'z' in data[c]['rk'] else gain_curve(c,g,'A50'); bk=max(cur,key=cur.get)
        P(f'   first {n} seeds: AUC {AUC(data[c]["y"],g):.5f} gain w.1 {cur[.1]*1e5:+.0f} w.05 {cur[.05]*1e5:+.0f} best {cur[bk]*1e5:+.0f}@{bk}')
fs=seeds(23,'u1'); 
if fs:
    g=member(23,fs); cur,a0=gain_curve(23,g,'base'); P(f'C23 u1(350 steps) seed0 only: AUC {AUC(data[23]["y"],g):.5f} gain w.1 {cur[.1]*1e5:+.0f} (compare u1b seed0 above)')
P('NOTE u2 (slate 3, bpr .5, drop .2): v21graph/logs/u2.log stops at step 450 (val25 .7475), no embeddings were saved -> cannot be scored.')
json.dump(out,open(D+'/heldout.json','w'),indent=1)
np.savez(D+'/graph_members.npz',**{f'G{c}':G[c].astype(np.float32) for c in G})
