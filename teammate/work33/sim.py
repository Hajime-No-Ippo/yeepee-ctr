import sys, json, numpy as np, polars as pl, os
os.chdir('/workspace/huawei/v9')
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score as A
V8='/workspace/huawei/v8'; EV=np.array([0.6662,0.1655,0.1682])
def z8(tag,c,SQ):
    b=np.load(f'ens/{tag}_b_c{c}_s0.npy')[:,1]; s=np.load(f'ens/{tag}_s_c{c}_s0.npy')[:,2]; return .8*(b+s)/2+.2*SQ
out=[]
for c in (25,23):
    v=pl.read_parquet(f'feat_v3/val{c}.parquet',columns=['rowid','eid','clicked','us_hlen']).join(pl.read_parquet(f'{V8}/feat_sqv/all500/val{c}.parquet',columns=['rowid','sq_logit']),on='rowid',how='left',maintain_order='left')
    y=v['clicked'].to_numpy(); SQ=v['sq_logit'].to_numpy(); eid=v['eid'].to_numpy(); hl=v['us_hlen'].to_numpy()
    st=pl.read_parquet(f'feat_tmB2/val{c}.parquet')['tmR_stale_h'].to_numpy(); bk=np.digitize(st,[24,48])
    sh=np.bincount(bk,minlength=3)/len(bk); pres=sh>0; tgt=np.where(pres,EV,0); tgt/=tgt.sum(); W=(tgt/np.where(pres,sh,1))[bk]
    hb=np.digitize(hl,[0.5,2.5,5.5,10.5])
    N=len(y); r8=rankdata(z8('trim',c,SQ))/N
    P={'noTT':rankdata(z8('noTT',c,SQ))/N}
    if os.path.exists(f'ens/p3like_b_c{c}_s0.npy'): P['p3like']=rankdata(np.load(f'ens/p3like_b_c{c}_s0.npy')[:,1])/N
    def st_(z):
        r=dict(auc=A(y,z),rw=A(y,z,sample_weight=W))
        for k,nm in enumerate(('s0_24','s24_48','s48p')):
            m=bk==k; r[nm]=A(y[m],z[m]) if m.sum()>1000 else np.nan
        for k,nm in enumerate(('h0','h1_2','h3_5','h6_10','h11p')):
            m=hb==k; r[nm]=A(y[m],z[m]) if y[m].sum()>30 else np.nan
        return r
    base=st_(r8); out.append(dict(cut=c,model='v8proxy(trim)',**{k:round(v,5) for k,v in base.items()}))
    df=pl.DataFrame({'eid':eid,'r8':r8})
    for nm,rp in P.items():
        sp=np.corrcoef(r8,rp)[0,1]
        out.append(dict(cut=c,model=nm+' alone',spearman_v8=round(sp,4),**{k:round(v,5) for k,v in st_(rp).items()}))
        for w in (0.1,0.2,0.3,0.4):
            bl=(1-w)*r8+w*rp; s=st_(bl)
            out.append(dict(cut=c,model=f'{nm} w{w}',**{'d_'+k:round(s[k]-base[k],5) for k in s}))
        # decomposition at w=0.3: within-event only / event-level only
        bl=0.7*r8+0.3*rp; d=pl.DataFrame({'eid':eid,'d':bl-r8}).with_columns(m=pl.col('d').mean().over('eid'))['m'].to_numpy()
        for lab,z in [('w0.3 within-event part only',bl-d),('w0.3 event-level part only',r8+d)]:
            s=st_(z); out.append(dict(cut=c,model=f'{nm} {lab}',**{'d_'+k:round(s[k]-base[k],5) for k in s}))
        # segment-swap: apply blend only in one segment (others keep v8 rank... use v8 score); measure total AUC gain
        for lab,m in [('stale24+',bk>=1),('fresh0-24',bk==0),('hist<=5',hb<=2),('hist>5',hb>2),('cold',hb==0)]:
            z=np.where(m,bl,r8); s=st_(z); out.append(dict(cut=c,model=f'{nm} w0.3 only on {lab}',d_auc=round(s['auc']-base['auc'],5),d_rw=round(s['rw']-base['rw'],5)))
    print('rows by hb',np.bincount(hb),'stale',np.bincount(bk,minlength=3))
for r in out: print(json.dumps(r))
json.dump(out,open('/workspace/huawei/teammate/work33/sim_results.json','w'))
