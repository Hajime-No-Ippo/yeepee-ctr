from lib import *
import sys
out=[]
for c in (25,23):
    F=load_cut(c); P=np.load(f'cache_prop_c{c}.npy'); names=json.load(open('prop_names.json')); y,z,u=F['y'],F['z'],F['user']
    n=len(y)//10; Z=z.reshape(n,10); Y=y.reshape(n,10); ey=Y.max(1); eu=u[::10]
    lse=np.log(np.exp(Z).sum(1)); sm=np.exp(Z-lse[:,None])/1 ; mx=Z.max(1); P1=P[::10]
    def ev(f): return A(ey,f)
    ctx=np.column_stack([lse,mx,Z.mean(1),Z.std(1),np.sort(Z,1)[:,-2]])
    def evlgb(X,rounds=150):
        o=np.zeros(n); 
        for tr,te in GroupKFold(5).split(X,ey,eu):
            m=lgb.train({**PI,'min_data_in_leaf':200,'num_leaves':7},lgb.Dataset(X[tr],ey[tr]),rounds); o[te]=m.predict(X[te],raw_score=True)
        return o
    r=dict(cut=c,ev_lse=ev(lse),ev_max=ev(mx))
    o1=evlgb(ctx); o2=evlgb(np.hstack([ctx,P1])); r.update(ev_ctx=ev(o1),ev_ctx_prop=ev(o2))
    # 2-stage global score: logit of P(click_j)=P(ev)*softmax_j
    lp=lambda o:(np.log(1/(1+np.exp(-o)))[:,None]+Z-lse[:,None]).ravel()
    for nm,o in (('ctx',o1),('ctx_prop',o2)):
        s=lp(o); r['glob_'+nm]=A(y,s)
    r['glob_z']=A(y,z)
    # oracle softmax with perfect event label
    r['glob_oracle_ev']=A(y,(np.log(np.clip(np.repeat(ey,10),1e-3,1))+ (Z-lse[:,None]).ravel()))
    print(json.dumps({k:(round(v,5) if isinstance(v,float) else v) for k,v in r.items()}),flush=True); open('results_p4.jsonl','a').write(json.dumps(r)+'\n')
