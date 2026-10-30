"""Step 1: error breakdown of v19 blend (saved C25 OOT preds; C23 dwbig only). <=2 threads. python diag1.py"""
import os
os.environ.update(OMP_NUM_THREADS='2',POLARS_MAX_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
import numpy as np, polars as pl, json, sys
from sklearn.metrics import roc_auc_score as AUC
from scipy.special import expit, logsumexp
H='/workspace/huawei'
def load_meta(cut):
    return pl.read_parquet(f'{H}/v12/feat_v3/val{cut}.parquet',columns=['rowid','eid','u','ts','clicked','it_imp','it_imp_rate','us_hlen','us_nclk_all','us_nev','us_cold','us_nohist','tm_stale','tm_hour','tm_tb','tm_dt','hist_in','hist_n_all','it_nclk_all','us_ctr'])
def blend(v,c):
    E=f'{H}/v19/ens'; L=lambda f,col:np.load(f'{E}/{f}.npy')[:,col]
    b=L(f'nm{v}s_c{c}_s0',0)
    if c!=25: return b,None
    d=L(f'nm{v}s_s_c{c}_s0',1); ml=np.mean([L(f'nm{v}smlp_c{c}_s{k}',3) for k in (0,1,2)],0); return b,(b+d+2*ml)/4
def within_auc(z,y):  # mean over clicked slates of (rank of the click among 10 -1)/9 with ties .5
    Z=z.reshape(-1,10); Y=y.reshape(-1,10); k=Y.sum(1)==1; Z=Z[k]; pos=Z[np.arange(len(Z)),Y[k].argmax(1)][:,None]
    return float(((Z<pos).sum(1)+0.5*((Z==pos).sum(1)-1)).mean()/9)
def decomp(z,y,tag):
    n=len(y); E=n//10; Z=z.reshape(E,10); Y=y.reshape(E,10); has=Y.max(1); R={}
    R['global']=AUC(y,z); R['within']=within_auc(z,y)
    Ec=int(has.sum()); N=int((1-y).sum()); R['pairs_same_share']=9*Ec/(Ec*N)
    # pos vs negs in clicked slates (other slates) / in no-click slates
    # slate-level has-click AUC
    p=expit(Z)
    for nm,s in (('max',Z.max(1)),('mean',Z.mean(1)),('sum_p',p.sum(1)),('lse',logsumexp(Z,1)),('1-prod(1-p)',1-np.prod(1-p,1))): R['slateAUC_'+nm]=AUC(has,s)
    # score from slate-level only / share only
    s_sum=p.sum(1); R['global_from_slate_sum_p_only']=AUC(y,np.repeat(s_sum,10))
    share=Z-logsumexp(Z,1,keepdims=True); R['global_from_softmax_share_only']=AUC(y,share.ravel())
    R['global_from_p_normalized(share*sum_p)']=AUC(y,(p/p.sum(1,keepdims=True)*s_sum[:,None]).ravel())  # = p
    # oracle within: clicked row gets slate's max score (multiset preserved)
    Zo=Z.copy(); k=has==1; ci=Y.argmax(1)
    mx=Z.max(1); tmp=Zo[np.arange(E),ci].copy(); am=Z.argmax(1)
    Zo[k,ci[k]]=mx[k]; Zo[k,am[k]]=np.where(am[k]==ci[k],mx[k],tmp[k])   # swap max and click
    R['oracle_within_global']=AUC(y,Zo.ravel())
    # oracle across: clicked slates shifted above no-click slates (within order kept)
    Za=Z+100.0*has[:,None]; R['oracle_across_global']=AUC(y,Za.ravel())
    Zb=Zo+100.0*has[:,None]; R['oracle_both_global']=AUC(y,Zb.ravel())
    # oracle across, softer: only add +1 logit to clicked slates
    for a in (0.5,1.0,2.0): R[f'across_shift_{a}']=AUC(y,(Z+a*has[:,None]).ravel())
    R['n_events']=E; R['click_event_rate']=float(has.mean()); R['ctr']=float(y.mean())
    return R
def seg_table(meta,z,y,col,edges=None,labels=None,qs=None,by_event=False):
    v=meta[col].to_numpy().astype(float)
    if qs is not None: edges=sorted(set(np.nanquantile(v,qs).tolist())); 
    b=np.digitize(np.nan_to_num(v,nan=-1e9),edges) if edges is not None else v.astype(int)
    out=[]; E=len(y)//10
    for g in np.unique(b):
        m=b==g
        if m.sum()<500 or y[m].sum()<20: continue
        yy=y[m]; zz=z[m]
        # within-slate AUC among the slates fully in segment (segment constant within event for user/time segs)
        ev=m.reshape(E,10).all(1); w=np.nan
        if ev.sum()>50: w=within_auc(z.reshape(E,10)[ev].ravel(),y.reshape(E,10)[ev].ravel())
        out.append(dict(seg=col,bin=int(g),lo=(edges[g-1] if edges is not None and g>0 else None),share=round(m.mean(),4),ctr=round(yy.mean(),4),auc=round(AUC(yy,zz),5),within=round(w,5) if w==w else None,
                        wdef=round(m.mean()*(AUC(y,z)-AUC(yy,zz)),5)))
    return out
if __name__=='__main__':
    res={}; seg={}
    for cut in (25,23):
        meta=load_meta(cut); y=meta['clicked'].to_numpy(); eid=meta['eid'].to_numpy().reshape(-1,10); assert (eid==eid[:,:1]).all()
        for v in 'BA':
            b,bl=blend(v,cut)
            for nm,z in (('dwbig',b),('blend',bl)):
                if z is None: continue
                key=f'C{cut}_{v}_{nm}'; res[key]=decomp(z,y,key); print(key,json.dumps({k:round(x,5) for k,x in res[key].items()}),flush=True)
        # segments on B blend (C25) / B dwbig (C23)
        z=blend('B',cut)[1] if cut==25 else blend('B',cut)[0]
        T=[]
        T+=seg_table(meta,z,y,'us_hlen',edges=[1,3,6,11,20])          # history length bins: 0,1-2,3-5,6-10,11-19,20
        T+=seg_table(meta,z,y,'us_nclk_all',qs=np.linspace(.1,.9,9))   # activity deciles
        T+=seg_table(meta,z,y,'us_cold',edges=[0.5]); T+=seg_table(meta,z,y,'us_nohist',edges=[0.5])
        T+=seg_table(meta,z,y,'it_imp',qs=np.linspace(.1,.9,9))        # item pop deciles
        T+=seg_table(meta,z,y,'tm_tb',edges=[0.5,1.5,2.5]); T+=seg_table(meta,z,y,'tm_hour',edges=[6,12,18])
        T+=seg_table(meta,z,y,'tm_stale',edges=[3600*6,86400,2*86400,3*86400])
        T+=seg_table(meta,z,y,'hist_in',edges=[0.5])
        seg[f'C{cut}']=T
        # slot-in-slate (row order inside event): no position column; test ordinal
        slot=np.tile(np.arange(10),len(y)//10); res[f'C{cut}_slot']=dict(single_auc=AUC(y,slot),ctr_by_slot=[round(float(y[slot==k].mean()),4) for k in range(10)])
    json.dump(dict(res=res,seg=seg),open('logs/diag1.json','w'),indent=1)
    for c,T in seg.items():
        print('==',c)
        for r in T: print(r)
