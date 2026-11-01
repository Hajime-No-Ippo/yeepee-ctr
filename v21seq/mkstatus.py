import glob,os,re,numpy as np,time
from evalseq import *
out=['# v21seq STATUS (auto-generated %s Europe/Dublin)\n'%time.strftime('%Y-%m-%d %H:%M'),
'ctl6 = noseq listwise MLP (v21list recipe + 6 epochs + BCE alpha .3); big = v21seq_big (trf history encoder, DIN attn, slate attn, 6ep, alpha .3, aug .3, frozen emb). base = .4z+.6L (z=v19 stack at C25, dwbig at C23; L=v21list s0 logit).\n']
def lg(f): return tologit(np.load(f)[:,-1])
def seeds(tag,var,c): return sorted(glob.glob(f'ens/{tag}_{var}_c{c}_s*.npy'),key=lambda f:int(re.search(r'_s(\d+)\.npy',f).group(1)))
for var in 'AB':
  for c in (23,25,26):
    tags=[t for t in ('ctl6','v21seq_big') if seeds(t,var,c)]
    if not tags: continue
    y=yv(c); out.append(f'\n## var {var} C{c}'+(' (TRANSDUCTIVE)' if var=='B' else '')+'\n')
    hasz=c in (23,25)
    if hasz:
        z=zbase(var,c); fl=sorted(glob.glob(f'{H}/v21diag/ens/v21list{var}_c{c}_s*.npy')); Ls=[tologit(np.load(f)[:,3]) for f in fl]; L=np.mean(Ls,0); base=.4*z+.6*L
        out.append(f'z {AUC(y,z):.5f} | v21list {var} ({len(fl)} seeds) {AUC(y,L):.5f} | base .4z+.6L {AUC(y,base):.5f}\n')
    out.append('\n| model | n | seed-avg AUC at k=1/2/4/8/all | last-seed-singles | blend .4z+.6M (all) | delta vs base |\n|---|---|---|---|---|---|\n')
    allP=[]
    for t in tags:
        P=[lg(f) for f in seeds(t,var,c)]; allP+=P; ks=[k for k in (1,2,4,8) if k<=len(P)]
        sat=' / '.join(f'{AUC(y,np.mean(P[:k],0)):.5f}' for k in ks)+(f' / {AUC(y,np.mean(P,0)):.5f}' if len(P) not in ks else '')
        sing=' '.join(f'{AUC(y,p):.5f}' for p in P[:8]); M=np.mean(P,0)
        bl=f'{AUC(y,.4*z+.6*M):.5f} | {AUC(y,.4*z+.6*M)-AUC(y,base):+.5f}' if hasz else '- | -'
        out.append(f'| {t} | {len(P)} | {sat} | {sing} | {bl} |\n')
    if len(tags)>1:
        M=np.mean(allP,0); out.append(f'| ALL | {len(allP)} | {AUC(y,M):.5f} | | '+(f'{AUC(y,.4*z+.6*M):.5f} | {AUC(y,.4*z+.6*M)-AUC(y,base):+.5f}' if hasz else '- | -')+' |\n')
out.append('\n## C27 (eval) prediction files\n'+'\n'.join(f'- {v} {t}: {len(seeds(t,v,27))} seeds' for v in 'AB' for t in ('ctl6','v21seq_big') if seeds(t,v,27))+'\n')
out.append('\n## Other results (C23, A)\n- slate correction (hlen/staleness logistic offset, cross-fit C25<->C23): -0.00005/-0.00006 (null)\n- GBDT lambdarank 0.79108 (500r) standalone; xendcg .79127; binary+slate-relative .79687 (vs binary .79679); blends with base: <= +0.00003\n- seq with learned item embeddings overfits (.7937->.7439); frozen embeddings: no gain over noseq in a like-for-like control (C23 big .79828 vs ctl .79824; C25 .81904 vs .81890)\n- blend weight w in (1-w)z+wM: optimum ~.7 (C23 .79907, C25 .81924), w=.6 within .00003\n')
import glob as _g
out.append('\n## Probes in out/ (sha256; ctl8 already copied to the Mac - do not touch)\n'+''.join(f'### {f}\n```\n'+open(f).read()+'```\n' for f in sorted(_g.glob('out/SHA256SUMS_*.txt'))))
open('STATUS.md','w').write(''.join(out)); print(''.join(out))
