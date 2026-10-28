from lib import *
import sys
cuts=[int(x) for x in sys.argv[1].split(',')]; seeds=[int(x) for x in sys.argv[2].split(',')]
names=json.load(open('tx_names.json')); fam={f:[i for i,n in enumerate(names) if n.startswith(f)] for f in ['f1','f2','f3','f4']}
sets=dict(f1=fam['f1'],f2=fam['f2'],f3=fam['f3'],f4=fam['f4'],all=list(range(len(names))))
for c in cuts:
    F=load_cut(c); T=np.load(f'cache_tx_c{c}.npy'); P=np.load(f'cache_prop_c{c}.npy'); y,z,u=F['y'],F['z'],F['user']
    for s in seeds:
        pc=oof(F['Xb'],y,z,u,s); 
        for nm,idx in list(sets.items())+[('all+prop',None)]:
            X=np.hstack([F['Xb'],T if idx is None else T[:,idx]]+([P] if idx is None else []))
            pt=oof(X,y,z,u,s); ci,sd=boot_delta(y,pc,pt,None,nb=150)
            r=dict(name='tx_'+nm,cut=c,seed=s,base=A(y,z),control=A(y,pc),test=A(y,pt),d_vs_ctrl=A(y,pt)-A(y,pc),ci=[float(ci[0]),float(ci[1])],sd=float(sd),nfeat=X.shape[1]-F['Xb'].shape[1])
            print(json.dumps({k:(round(v,5) if isinstance(v,float) else v) for k,v in r.items()}),flush=True); open('results_tx.jsonl','a').write(json.dumps(r)+'\n')
