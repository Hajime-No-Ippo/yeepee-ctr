"""make_report.py: reads search_out.pkl + oof_c*.npz (via blend_oof.build) -> REPORT.md, best_weights.json. Reproducible, no eval file written."""
import sys, json, pickle, numpy as np
sys.path.insert(0,'/workspace/huawei/cb_blend')
import blend_oof as b
d=b.build(); P=pickle.load(open(f'{b.OUT}/search_out.pkl','rb')); S=P['S']; log=P['log']
y={c:d[c]['y'] for c in d}
base={c:b.AUC(y[c],.4*d[c]['rk']['z']+.6*d[c]['rk']['M']) for c in (23,25)}
base[26]=None
def fmt(x): return '-' if x is None else f'{x:.5f}'
def dl(x,c): return '-' if (x is None or base[c] is None) else f'{(x-base[c])*1e5:+.1f}'
L=[]
# ---- candidate table: held-out only (fit on the other fold)
L.append('## 3. Held-out candidates (weights fitted on ONE fold, scored on the OTHER). delta in 1e-5 AUC vs .4z+.6M (rank space) on the same fold\n')
L.append('| member set | procedure | weights (order = set) | fit C25 -> held-out C23 | delta | fit C23 -> held-out C25 | delta | mean held-out delta (1e-5) |\n|---|---|---|---|---|---|---|---|')
byk={}
for r in S:
    if not r['held']: continue
    lab=r['label']; proc=lab.replace('fit C23','').replace('fit C25','').strip() or 'plain fit'
    byk.setdefault((r['set'],proc),{})[list(r['held'])[0]]=r
rows=[]
for (st,proc),v in byk.items():
    if 23 in v and 25 in v:
        a23=v[23]['held'][23]; a25=v[25]['held'][25]; md=((a23-base[23])+(a25-base[25]))/2*1e5
        rows.append((md,st,proc,v[23],v[25],a23,a25))
rows.sort(key=lambda r:-r[0]); 
for md,st,proc,r23,r25,a23,a25 in rows:
    w23=','.join(f'{x:.2f}' for x in r23['w']); w25=','.join(f'{x:.2f}' for x in r25['w'])
    L.append(f'| {st} | {proc} | C25-fit {w23} / C23-fit {w25} | {a23:.5f} | {dl(a23,23)} | {a25:.5f} | {dl(a25,25)} | {md:+.1f} |')
# fixed references
L.append('\n## Fixed (unfitted) references, rank space\n\n| candidate | C23 | delta | C25 | delta |\n|---|---|---|---|---|')
for r in P['res']:
    L.append(f"| {r['cand']} ({r['w']}) | {fmt(r['auc'].get(23))} | {dl(r['auc'].get(23),23)} | {fmt(r['auc'].get(25))} | {dl(r['auc'].get(25),25)} |")
L.append('\nRaw (logit-space) reproduction of Grok Bot numbers: '+', '.join(f'{k} {v:.5f}' for k,v in log['rawrep'].items()))
open('/tmp/_L.txt','w').write('\n'.join(L))
print('\n'.join(L))
# ---- member table
print('\n## member standalone + rho vs M')
for c in (23,25,26):
    for m in d[c]['rk']:
        rho=b.spearmanr(d[c]['rk'][m][::3],d[c]['rk']['M'][::3])[0]; print(c,m,round(b.AUC(y[c],d[c]['rk'][m]),5),round(rho,4))
