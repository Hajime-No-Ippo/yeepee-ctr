import numpy as np, json
from common import *
out={}
for c in (25,23):
    m=meta(c); y=m['clicked'].to_numpy(); z=z12(c); n=len(y)//10
    assert (m['eid'].to_numpy().reshape(n,10)==m['eid'].to_numpy().reshape(n,10)[:,:1]).all()
    Y=y.reshape(n,10); Z=z.reshape(n,10); yev=Y.max(1); P=y.sum(); N=len(y)-P
    lse=np.log(np.exp(Z).sum(1)); soft=Z-lse[:,None]
    r=dict(cut=c,events=n,pos=int(P),ev_click_rate=float(yev.mean()),auc=A(y,z))
    # pair counts
    npos_ev=Y.sum(1); within=int((npos_ev*(10-npos_ev)).sum()); total=int(P*N)
    r['within_pair_share']=within/total
    # within-event AUC (pooled over click events)
    ce=yev==1
    r['auc_within_pooled']=A(Y[ce].ravel(),soft[ce].ravel())   # same as z within-event since subtracting const
    # mean per-event AUC
    r['auc_global_soft']=A(y,soft.ravel())
    r['auc_event_lse']=A(yev,lse); r['auc_event_pany']=A(yev,np.log1p(-np.prod(1-1/(1+np.exp(-Z)),1)*0.9999999))
    r['auc_global_lse_only']=A(y,np.repeat(lse,10))
    # between-event split: negatives from no-click events vs click events(other events)
    # AUC of pos vs negs in no-click events
    pos_s=z[y==1]; 
    negs_nc=Z[~ce].ravel(); negs_c=Z[ce].ravel()  # includes pos in ce -> need only negatives
    negs_c=Z[ce][Y[ce]==0]
    from scipy.stats import rankdata
    def auc2(p,q): 
        a=np.concatenate([p,q]); r_=rankdata(a); return (r_[:len(p)].sum()-len(p)*(len(p)+1)/2)/(len(p)*len(q))
    r['auc_pos_vs_negNoClickEv']=auc2(pos_s,negs_nc); r['auc_pos_vs_negClickEv(incl same)']=auc2(pos_s,negs_c)
    r['n_neg_noclick']=len(negs_nc); r['n_neg_click']=len(negs_c)
    # oracles
    r['oracle_event_label']=A(y,np.repeat(yev,10)*100+z)     # perfect any-click, model within
    r['oracle_event_label_only']=A(y,np.repeat(yev,10))
    r['oracle_within']=A(y,np.where(y==1,1,0)*100+ (np.repeat(lse,10))*0 + np.repeat(lse,10)*0 + z*0+  np.repeat(lse,10)) # perfect within-ranking (clicked row top of event), event score lse as before
    # perfect within, event score = softmax mass: row score = lse + big*y? use lse + y*delta (delta modest = within gap)
    for dl in (1,2,3):
        r[f'oracle_within_delta{dl}']=A(y,np.repeat(lse,10)+dl*y*(np.repeat(yev,10)))
    # oracle on no-click events: those would be known as no-click, push down by k
    for k in (0.5,1,2):
        r[f'oracle_noclick_shift{k}']=A(y,z-k*np.repeat(1-yev,10))
    # score mixing: z + a*lse
    for a in (-0.5,-0.25,0.25,0.5):
        r[f'z+{a}lse']=A(y,z+a*np.repeat(lse,10))
    out[c]=r; print(json.dumps({k:(round(v,5) if isinstance(v,float) else v) for k,v in r.items()}),flush=True)
json.dump(out,open('decomp.json','w'))
