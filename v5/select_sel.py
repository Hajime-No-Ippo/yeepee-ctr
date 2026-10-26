import json,numpy as np,sys
R=[json.loads(l) for l in open('out/con_sweep.jsonl')]
seen={};[seen.__setitem__(r['name'],r) for r in R]
R=sorted(seen.values(),key=lambda r:-r['within'])
ref=np.load('out/cs_x_con05_k32_tfidf05.npy')
for r in R[:12]: print(r['name'],r['within'],r['auc_z'],r['corr_ref'])
sel=[]
for r in R:
    if r['name']=='x_con05_k32_tfidf05' or abs(r['corr_ref'])>=0.9: continue
    v=np.load(f"out/cs_{r['name']}.npy")
    if all(abs(np.corrcoef(v,np.load(f"out/cs_{s['name']}.npy"))[0,1])<0.9 for s in sel): sel.append(r)
    if len(sel)==4: break
print('SEL',[(s['name'],s['within'],s['corr_ref']) for s in sel])
if len(sys.argv)>1:
    json.dump([[s['k'],s['w'],s['beta'],s['alpha'],s['name']] for s in sel],open('out/SEL.json','w'))
