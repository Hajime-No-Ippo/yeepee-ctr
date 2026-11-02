import pickle,csv,numpy as np
st=pickle.load(open('eval_stats.pkl','rb')); rows=pickle.load(open('eval_table.pkl','rb')); LG=pickle.load(open('local_gain.pkl','rb')); HO=pickle.load(open('heldout.pkl','rb'))['HO']
Rl=LG['R']; WG=LG['WG']
with open('inventory.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['member','family','n_seeds_avg','order','rho_vs_EN_aligned','rho_vs_EN_if_shuffled(ctrl~0)'])
    for k,fam,ns,r,sh in st['INV']: w.writerow([k,fam,ns,'merged.parquet rowid order (=eval.parquet=evaluation.csv); v21seq/xgb8/graph arrays re-joined by rowid',round(r,4),round(sh,4)])
out=[]
for r in rows:
    k=r['member']; d=dict(r); L=Rl.get(k,{})
    for c in (23,25,26):
        d[f'locAUC_c{c}']=L.get('aucs',{}).get(c); d[f'rhoM_c{c}']=L.get('rhoM',{}).get(c)
        if c in L.get('curve',{}): d[f'bestgain_c{c}_e4']=float(L['curve'][c].max()*1e4); d[f'bestw_c{c}']=WG[int(np.argmax(L['curve'][c]))]
    d.update({f'HO_{a}':float(b) for a,b in HO.get(k,{}).items()}); out.append(d)
for k in ('cat_numte(v21cat)','ref484(v20)','gf_num_s3(v21gfeat)','gf_numg_s3(v21gfeat)'):
    L=Rl[k]; d=dict(member=k,rho_EN=None,note='NO eval array'); 
    for c in (23,25,26):
        if c in L['aucs']: d[f'locAUC_c{c}']=L['aucs'][c]; d[f'rhoM_c{c}']=L['rhoM'][c]; d[f'bestgain_c{c}_e4']=float(L['curve'][c].max()*1e4); d[f'bestw_c{c}']=WG[int(np.argmax(L['curve'][c]))]
    d.update({f'HO_{a}':float(b) for a,b in HO.get(k,{}).items()}); out.append(d)
keys=sorted({a for d in out for a in d},key=lambda s:(s!='member',s)); 
with open('results.csv','w',newline='') as f:
    w=csv.DictWriter(f,keys); w.writeheader(); [w.writerow(d) for d in out]
for d in out:
    f=lambda x:'-' if x is None or x!=x else f'{x:.3f}'
    print(f"{d['member']:22s} rEN {f(d.get('rho_EN'))} rS1 {f(d.get('rho_S1'))} rsd {f(d.get('resid_sd'))} | loc gain e-4 c23 {f(d.get('bestgain_c23_e4'))} c25 {f(d.get('bestgain_c25_e4'))} c26 {f(d.get('bestgain_c26_e4'))} | HO {f(d.get('HO_g23_w25') if 'HO_g23_w25' in d else d.get('HO_g26_w25'))}/{f(d.get('HO_g25_w23'))} | rhoM25 {f(d.get('rhoM_c25'))}")
