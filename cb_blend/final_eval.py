"""final_eval.py: evaluate the shortlisted regularised candidates (fixed weights, no fitting) + uncertainty by user-hash subsets; C26 check; writes final_eval.json"""
import sys, json, numpy as np
sys.path.insert(0,'/workspace/huawei/cb_blend'); import blend_oof as b
d=b.build(); G=8
def sc(c,W): return sum(w*d[c]['rk'][m] for m,w in W.items())
CANDS={'base .4z+.6M':{'z':.4,'M':.6},
 'eq4 (z,M,big,D) .25 each':{'z':.25,'M':.25,'big':.25,'D':.25},
 'eq5 (z,M,big,D,ref) .2 each':{'z':.2,'M':.2,'big':.2,'D':.2,'ref':.2},
 'eq4b (z,M,big,ref) .25 each':{'z':.25,'M':.25,'big':.25,'ref':.25},
 'eq3 (z,M,big) 1/3':{'z':1/3,'M':1/3,'big':1/3},
 'eq4c (z,M,D,ref) .25 each':{'z':.25,'M':.25,'D':.25,'ref':.25},
 'eq3d (z,A=M+big,D) .3/.3/.4':{'z':.3,'M':.15,'big':.15,'D':.4},
 'sh5 (50% mean-of-folds fit + 50% equal)':None,
 'eq4+graph .1 (z,M,big,D .225 each, graph .1)':{'z':.225,'M':.225,'big':.225,'D':.225,'graph':.1},
 'eq4+cat .1 (z,M,big,D .225 each, cat .1)':{'z':.225,'M':.225,'big':.225,'D':.225,'cat':.1},
 'eq4+cat .05+graph .1':{'z':.2125,'M':.2125,'big':.2125,'D':.2125,'cat':.05,'graph':.1},
 'base+graph .1 (.36z+.54M+.1graph)':{'z':.36,'M':.54,'graph':.1}}
# mean-of-folds fit for sh5 (read from search pkl)
import pickle; S=pickle.load(open(f'{b.OUT}/search_out.pkl','rb'))['S']
mf=[r for r in S if r['set']=='z,M,big,D,ref' and r['label'].startswith('mean-of-folds')][0]; names=mf['names']; w=.5*mf['w']+.5/len(names)
CANDS['sh5 (50% mean-of-folds fit + 50% equal)']=dict(zip(names,[float(x) for x in w])); 
out={}
EQ4={'z':.25,'M':.25,'big':.25,'D':.25}
base={c:sc(c,CANDS['base .4z+.6M']) for c in (23,25)}
for k,W in CANDS.items():
    r=dict(W={m:round(x,3) for m,x in W.items()})
    for c in (23,25):
        s=sc(c,W); a=b.AUC(d[c]['y'],s); m,sd=b.user_group_delta(d,c,s,base[c],G)
        r[f'C{c}']=a; r[f'dC{c}']=a-b.AUC(d[c]['y'],base[c]); r[f'dC{c}_ug']=[m,sd]; r[f'rhoM_C{c}']=float(b.spearmanr(s[::3],d[c]['rk']['M'][::3])[0])
        if 'graph' in W or 'cat' in W: m2,sd2=b.user_group_delta(d,c,s,sc(c,EQ4),G); r[f'dC{c}_vs_eq4']=a-b.AUC(d[c]['y'],sc(c,EQ4)); r[f'dC{c}_vs_eq4_ug']=[m2,sd2]
    out[k]=r; print(k,{x:(round(v,5) if isinstance(v,float) else v) for x,v in r.items()},flush=True)
# ref effect: eq5 vs eq4 on user subsets
for c in (23,25):
    a=sc(c,CANDS['eq5 (z,M,big,D,ref) .2 each']); bb=sc(c,CANDS['eq4 (z,M,big,D) .25 each']); m,sd=b.user_group_delta(d,c,a,bb,G)
    print('ref effect eq5-eq4 C',c,'delta',b.AUC(d[c]['y'],a)-b.AUC(d[c]['y'],bb),'ug mean/sd',m,sd); out[f'ref_effect_C{c}']=dict(delta=b.AUC(d[c]['y'],a)-b.AUC(d[c]['y'],bb),ug=[m,sd])
    a=sc(c,CANDS['eq4b (z,M,big,ref) .25 each']); bb=sc(c,CANDS['eq3 (z,M,big) 1/3']); out[f'ref_effect_nodcn_C{c}']=b.AUC(d[c]['y'],a)-b.AUC(d[c]['y'],bb); print('ref effect (no D) C',c,out[f'ref_effect_nodcn_C{c}'])
# C26 (members M,big only): M/big mix
y=d[26]['y']; r26={}
for lab,(wm,wb) in {'M only':(1,0),'big only':(0,1),'50/50':(.5,.5),'M .33 big .67':(1/3,2/3),'A_all(logit mean)':None}.items() if False else []: pass
for lab,s in {'M only':d[26]['rk']['M'],'big only':d[26]['rk']['big'],'M/big 50-50':.5*d[26]['rk']['M']+.5*d[26]['rk']['big'],'M .33/big .67':d[26]['rk']['M']/3+d[26]['rk']['big']*2/3,'A_all (logit mean)':d[26]['rk']['A']}.items(): r26[lab]=b.AUC(y,s)
out['C26']=r26; print('C26',r26)
# C23/C25 ctl6-vs-big weights inside z,M,big (held-out gist): M/big 50-50 vs M only, within base .4z+.6x
for c in (23,25):
    for lab,x in {'M':d[c]['rk']['M'],'M/big 50-50':.5*d[c]['rk']['M']+.5*d[c]['rk']['big']}.items(): print(c,'.4z+.6',lab,round(b.AUC(d[c]['y'],.4*d[c]['rk']['z']+.6*x),5))
json.dump(out,open(f'{b.OUT}/final_eval.json','w'),indent=1)
