#!/usr/bin/env python
"""v21roll/rollval.py -- rolling (walk-forward) date validation for the TechArena CTR pipeline.  No upload, no eval rows, no eval labels.
Subcommands
  days                                   day table (events/slates/users/CTR) + first-event fold tables
  plan                                   which (train tables -> test tables) predictions a candidate must supply
  fit   --variant V [--folds 3,4,5,6]    fast LightGBM (cached features) per fold -> preds/{V}_M{d}_{table}.npy  
  score --names a,b,c [--ref a] [--pred-dir D]   AUC per fold (plain + staleness-matched), weighted means, bootstrap deltas
Fold d (d=3..6 = Jan 23..26): train = first-event-per-user tables f1..f{d-1} (features at each table's own cutoff; labels all < Jan 20+d days),
test = table f{d} (first event/user of day d, features at cutoff = start of day d, so staleness 0-24h).  M_d is the model trained on f1..f{d-1}.
Staleness-matched version (SM_d, d=4,5,6): each day-d event is scored with the STALEST cached feature snapshot that contains it
(val23 = cutoff Jan23, val25 = cutoff Jan25; otherwise f{d} fresh) and reweighted so the mix is 66.6% fresh / 16.6% 24-48h / 16.8% 48h+ (eval's mix; missing bins' mass goes to the available stale bins).
Also from the same models: C23 = M3 on val23 (train f1,f2), C25 = M5 on val25 (train f1..f4) -- the existing cuts.
Custom pipeline hooks: --pred-dir D with files {name}_M{d}_{table}.npy (float scores, row order of the cached table; `plan` lists them).
"""
import sys, os, json, time, glob, re, argparse, gc
import numpy as np, polars as pl
H='/workspace/huawei'; R=f'{H}/v21trend'; RC=f'{H}/v21roll/cache'; J20=1737331200; DAY=86400
CACHE=f'{H}/v9d/val/cache_mx'
BASE5=('feat_lr2','feat_tt','feat_pp','feat_sqC','feat_dw')
VARIANTS={'v12':BASE5,'coA':BASE5+('feat_Ap',),'coB':BASE5+('feat_Bp',)}
for _n,_e in (('trI',('feat_trI',)),('trU',('feat_trU',)),('trIU',('feat_trI','feat_trU'))):
    VARIANTS['coA_'+_n]=BASE5+('feat_Ap',)+_e; VARIANTS['coB_'+_n]=BASE5+('feat_Bp',)+_e
DROPRE=re.compile(r'^sq_(pdot|pcos|mdot|mcos)$')
TESTS={3:['f3','val23'],4:['f4','val23'],5:['f5','val25','val23'],6:['f6','val25','val23']}
PUBLIC={('v12','coA'):0.818270-0.818206,('coA','coB'):0.818302-0.818270,('v12','coB'):0.818302-0.818206}  # p33_w80 probes (v12 .818206, A .818270, B .818302)
def extdir(sub):
    for b in ('v21trend','v17cooc','v12x','v12','v19'):
        p=f'{H}/{b}/{sub}'
        if os.path.isdir(p): return p
    raise FileNotFoundError(sub)
def cachefp(n):
    n2={'f6':'f6'}.get(n,n); g=glob.glob(f'{CACHE}/{n2}_*.parquet'); g=[x for x in g if not x.endswith('.json')]; assert len(g)==1,(n,g); return g[0]
def meta(n):
    p=f'{RC}/meta_{n}.parquet'
    if os.path.exists(p): return pl.read_parquet(p)
    m=pl.read_parquet(cachefp(n),columns=['rowid','eid','u','ts','clicked','tm_stale']); m.write_parquet(p); return m
def load_X(n, extra, negfrac=1.0, seed=0, with_y=True):
    """feature matrix float32 for table n (optionally keeping all positives + negfrac of negatives), same rows for every variant"""
    fp=cachefp(n); fl=json.load(open(fp+'.json')); t=pl.read_parquet(fp)
    if negfrac<1.0:
        y=t['clicked'].to_numpy(); rng=np.random.default_rng(1000*seed+int(re.sub(r'\D','',n)[-2:] or 0)); keep=(y==1)|(rng.random(len(y))<negfrac); t=t.filter(pl.Series(keep))
    for sub in extra:
        if sub in ('feat_lr2','feat_tt','feat_pp'): continue
        e=pl.read_parquet(extdir(sub)+f'/{n}.parquet'); nc=[c for c in e.columns if c!='rowid' and c not in fl]
        t=t.join(e.select(['rowid']+nc),on='rowid',how='left',maintain_order='left'); fl=fl+nc; del e
    fl=[c for c in fl if not DROPRE.match(c)]
    y=t['clicked'].to_numpy().astype(np.float32); rid=t['rowid'].to_numpy()
    X=t.select(fl).to_numpy(); assert X.dtype==np.float32; del t; gc.collect()
    return X,y,rid,fl
LGBP=dict(objective='binary',learning_rate=0.06,num_leaves=63,min_data_in_leaf=200,feature_fraction=0.5,bagging_fraction=0.8,bagging_freq=1,lambda_l2=30.0,max_bin=63,force_col_wise=True,verbose=-1)
NEGF=0.35; ROUNDS=300
def getvar(v):
    if v in VARIANTS: return VARIANTS[v]
    b,*xs=v.split('+')
    if b not in VARIANTS: b0,x0=b.rsplit('_',1); b=b0; xs=[x0]+xs
    return VARIANTS[b]+tuple('feat_'+x for x in xs)
def cmd_fit(a):
    import lightgbm as lgb
    extra=getvar(a.variant); log=open(f'{R}/logs/fit_log.jsonl','a')
    for d in [int(x) for x in a.folds.split(',')]:
        t0=time.time(); Xs=[];ys=[]
        for k in range(1,d):
            X,y,_,fl=load_X(f'f{k}',extra,NEGF,a.seed); Xs.append(X); ys.append(y)
        X=np.concatenate(Xs); y=np.concatenate(ys); del Xs; gc.collect(); tl=time.time()-t0
        P=dict(LGBP,num_threads=a.threads,seed=42+a.seed,learning_rate=a.lr)
        t1=time.time(); ds=lgb.Dataset(X,y,feature_name=fl,free_raw_data=True,params=dict(max_bin=63,verbose=-1)).construct(); del X; gc.collect(); td=time.time()-t1
        t1=time.time(); mdl=lgb.train(P,ds,a.rounds); tf=time.time()-t1; del ds; gc.collect()
        aucs={}
        from sklearn.metrics import roc_auc_score
        for tb in TESTS[d]:
            Xt,yt,rid,fl2=load_X(tb,extra,1.0); assert fl2==fl
            s=mdl.predict(Xt,raw_score=True).astype(np.float32)+np.float32(np.log(NEGF)); del Xt; gc.collect()
            np.save(f'{R}/preds/{a.variant}{"_s%d"%a.seed if a.seed else ""}_M{d}_{tb}.npy',s); aucs[tb]=round(float(roc_auc_score(yt,s)),5)
        msg=dict(variant=a.variant,fold=d,n_train=len(y),nf=len(fl),load_s=round(tl),dataset_s=round(td),fit_s=round(tf),total_s=round(time.time()-t0),rounds=a.rounds,threads=a.threads,auc=aucs)
        print(json.dumps(msg),flush=True); log.write(json.dumps(msg)+'\n'); log.flush(); del mdl; gc.collect()

def slate_agg(n,extra):
    """slate-level aggregates (mean/max/sd over the 10 rows) of all row features of table n; cached"""
    p=f'{RC}/slate_{"_".join(e[5:] for e in extra[5:]) or "base"}_{n}.npz'
    if os.path.exists(p): d=np.load(p); return d['F'],d['h']
    X,y,rid,fl=load_X(n,extra,1.0); N=len(y); assert N%10==0
    e=meta(n)['eid'].to_numpy(); E=N//10; assert (e.reshape(E,10)==e.reshape(E,10)[:,:1]).all()
    Xr=X.reshape(E,10,-1); F=np.concatenate([np.nanmean(Xr,1),np.nanmax(Xr,1),np.nanstd(Xr,1)],1).astype(np.float32); del X,Xr; gc.collect()
    h=y.reshape(E,10).max(1); np.savez(p,F=F,h=h); return F,h
def lse10(z):
    z=z.reshape(-1,10); m=z.max(1,keepdims=True); return (m[:,0]+np.log(np.exp(z-m).sum(1)))
def cmd_slate(a):
    """slate-correction layer (v21slate recipe): score = z - c*lse(z) + a*P, P = out-of-time slate-has-click LightGBM (step3.py recipe, trained on f1..f{d-1} aggregates). Writes preds/{V}s_M{d}_{tb}.npy"""
    import lightgbm as lgb
    extra=getvar(a.variant); out=a.variant+'s'; t0=time.time(); log=open(f'{R}/logs/fit_log.jsonl','a')
    from sklearn.metrics import roc_auc_score
    for d in [int(x) for x in a.folds.split(',')]:
        t1=time.time(); T=[slate_agg(f'f{k}',extra) for k in range(1,d)]; X=np.concatenate([t[0] for t in T]); y=np.concatenate([t[1] for t in T]); del T
        keep=[i for i in range(X.shape[1]) if np.nanstd(X[:,i])>0]; X=X[:,keep]
        P=dict(objective='binary',learning_rate=0.03,num_leaves=31,min_data_in_leaf=100,feature_fraction=0.3,bagging_fraction=0.8,bagging_freq=1,lambda_l2=30,max_bin=63,verbose=-1,num_threads=a.threads,seed=42)
        m=lgb.train(P,lgb.Dataset(X,y,params=dict(max_bin=63,verbose=-1)),a.srounds); aucs={}
        for tb in TESTS[d]:
            F,h=slate_agg(tb,extra); p=m.predict(F[:,keep],raw_score=True); z=np.load(f'{R}/preds/{a.variant}_M{d}_{tb}.npy').astype(np.float64)
            s2=z-a.c*np.repeat(lse10(z),10)+a.a*np.repeat(p,10); np.save(f'{R}/preds/{out}_M{d}_{tb}.npy',s2.astype(np.float32))
            yy=meta(tb)['clicked'].to_numpy(); aucs[tb]=[round(float(roc_auc_score(yy,z)),5),round(float(roc_auc_score(yy,s2)),5),round(float(roc_auc_score(h,p)),4)]
        msg=dict(variant=out,fold=d,n_slates_train=len(y),srounds=a.srounds,total_s=round(time.time()-t1),auc_base_corr_slateAUC=aucs); print(json.dumps(msg),flush=True); log.write(json.dumps(msg)+'\n'); log.flush()
def cmd_days(a):
    TE=pl.read_parquet(f'{H}/lite/train_events.parquet',columns=['eid','user','ts']); TR=pl.read_parquet(f'{H}/lite/train_rows.parquet',columns=['eid','clicked'])
    ck=TR.group_by('eid').agg(pl.col('clicked').sum().alias('clk'),pl.len().alias('n'))
    t=TE.join(ck,on='eid').with_columns(((pl.col('ts')-J20)//DAY).alias('day'))
    g=t.group_by('day').agg(pl.len().alias('slates'),pl.col('n').sum().alias('rows'),pl.col('user').n_unique().alias('users'),pl.col('clk').sum().alias('clicks'),(pl.col('clk')>0).mean().alias('slates_w_click')).sort('day')
    g=g.with_columns((pl.col('clicks')/pl.col('rows')).alias('ctr'),(pl.col('day')+20).alias('jan'))
    print(g.select(['jan','day','slates','rows','users','clicks','ctr','slates_w_click']))
    print('first-event-per-user fold tables:')
    for n in ['f1','f2','f3','f4','f5','f6','val23','val25','val26']:
        m=meta(n); ev=m.group_by('eid').agg(pl.col('tm_stale').first(),pl.col('clicked').sum().alias('c'))
        st=ev['tm_stale'].to_numpy()/3600; print(n,'slates',len(ev),'rows',len(m),'ctr',round(m['clicked'].mean(),4),'stale<24h/24-48/48+',[round(float(x),3) for x in ((st<24).mean(),((st>=24)&(st<48)).mean(),(st>=48).mean())])
def cmd_plan(a):
    for d,tb in TESTS.items(): print(f'M{d}: train f1..f{d-1}  -> predict {tb}   files {{name}}_M{d}_{{table}}.npy')
# ---------------- scoring ----------------
def wauc(s,y,w):
    o=np.argsort(s,kind='stable'); y=y[o]; w=w[o]; wp=w*y; wn=w*(1-y); cn=np.cumsum(wn)
    return float((wp*(cn-0.5*wn)).sum()/(wp.sum()*wn.sum()))
class Fold:
    """test set for one fold d, kind plain/sm: arrays y,w,event index, and a row-gather per predictions dict"""
    def __init__(s,d,kind):
        s.d=d; s.kind=kind; mf=meta(f'f{d}'); s.rows=[]  # list of (table, row index array) in output order
        day0=J20+d*DAY
        if kind=='plain':
            s.parts=[(f'f{d}',np.arange(len(mf)))]; s.bin=np.zeros(len(mf),int)
        else:
            ev_f=mf.with_row_index('r').group_by('eid',maintain_order=True).agg(pl.col('r'))
            fresh_ev=set(ev_f['eid'].to_list()); assigned={}
            for tb,c in (('val23',3),('val25',5)):   # stalest first
                if c>=d: continue
                m=meta(tb).with_row_index('r'); m=m.filter((pl.col('ts')>=day0)&(pl.col('ts')<day0+DAY))
                evs=m.group_by('eid',maintain_order=True).agg(pl.col('r'),pl.col('tm_stale').first())
                for e,r,st in zip(evs['eid'].to_list(),evs['r'].to_list(),evs['tm_stale'].to_list()):
                    if e in fresh_ev and e not in assigned: assigned[e]=(tb,r,0 if st<DAY else 1 if st<2*DAY else 2)
            parts={}; binv=[]
            for e,r in zip(ev_f['eid'].to_list(),ev_f['r'].to_list()):
                if e in assigned: tb,rr,b=assigned[e]
                else: tb,rr,b=f'f{d}',r,0
                parts.setdefault(tb,[]).append(np.array(rr)); binv.append((tb,len(rr),b))
            s.parts=[(tb,np.concatenate(v)) for tb,v in parts.items()]
            bins=[]
            for tb,idx in s.parts:
                m=meta(tb); st=m['tm_stale'].to_numpy()[idx]; bins.append(np.where(st<DAY,0,np.where(st<2*DAY,1,2)))
            s.bin=np.concatenate(bins)
        s.y=np.concatenate([meta(tb)['clicked'].to_numpy()[idx] for tb,idx in s.parts]).astype(np.float64)
        eid=np.concatenate([meta(tb)['eid'].to_numpy()[idx] for tb,idx in s.parts]); s.ev=np.unique(eid,return_inverse=True)[1]; s.nev=s.ev.max()+1
        tgt=np.array([0.6662,0.1655,0.1682]); present=np.array([(s.bin==b).any() for b in range(3)])
        if kind=='plain': s.w=np.ones(len(s.y)); s.mix=[1.0,0,0]
        else:
            st_m=tgt[1:]*present[1:]; fresh=tgt[0]; stale_mass=1-fresh
            mix=np.zeros(3); mix[0]=fresh; 
            if st_m.sum()>0: mix[1:]=stale_mass*st_m/st_m.sum()
            else: mix[0]=1
            cnt=np.array([(s.bin==b).sum() for b in range(3)]).clip(1); s.w=mix[s.bin]/cnt[s.bin]*len(s.y); s.mix=mix.round(4).tolist(); s.counts=cnt.tolist()
    def gather(s,P):
        return np.concatenate([P[tb][idx] for tb,idx in s.parts])
def load_preds(name,pdir):
    P={}
    for d in TESTS:
        P[d]={tb:np.load(f'{pdir}/{name}_M{d}_{tb}.npy').astype(np.float64) for tb in TESTS[d] if os.path.exists(f'{pdir}/{name}_M{d}_{tb}.npy')}
    return P
def weights(K,scheme):
    k=np.arange(1,K+1)
    return {'mean':np.ones(K),'lin':k.astype(float),'exp':2.0**(k-1)}[scheme]/{'mean':K,'lin':k.sum(),'exp':(2.0**(k-1)).sum()}[scheme]
def cmd_score(a):
    names=a.names.split(','); ref=a.ref or names[0]; pdir=a.pred_dir or f'{R}/preds'; B=a.boot; rng=np.random.default_rng(0)
    Pr={n:load_preds(n,pdir) for n in names}
    sets={}  # label -> (fold object, accessor)
    for d in (3,4,5,6): sets[f'plain{d}']=(Fold(d,'plain'),d)
    for d in (4,5,6): sets[f'sm{d}']=(Fold(d,'sm'),d)
    class Cut:  # C23 / C25 full val tables
        def __init__(s,tb,d):
            m=meta(tb); s.parts=[(tb,np.arange(len(m)))]; s.y=m['clicked'].to_numpy().astype(float); s.w=np.ones(len(s.y)); s.ev=np.unique(m['eid'].to_numpy(),return_inverse=True)[1]; s.nev=s.ev.max()+1
        def gather(s,P): return np.concatenate([P[tb][i] for tb,i in s.parts])
    sets['C23']=(Cut('val23',3),3); sets['C25']=(Cut('val25',5),5)
    res={}; draws={}
    cnts=[rng.poisson(1.0,size=(B,sets[k][0].nev)).astype(np.float64) for k in sets]
    for (k,(F,d)),cn in zip(sets.items(),cnts):
        sc={n:F.gather(Pr[n][d]) for n in names}
        res[k]={n:wauc(sc[n],F.y,F.w) for n in names}
        # bootstrap: shared event multiplicities; sort once per variant
        dr={}
        for n in names:
            o=np.argsort(sc[n],kind='stable'); y=F.y[o]; w=F.w[o]; ev=F.ev[o]; out=np.empty(B)
            for b in range(B):
                ww=w*cn[b][ev]; wp=ww*y; wn=ww*(1-y); c=np.cumsum(wn); out[b]=(wp*(c-0.5*wn)).sum()/(wp.sum()*wn.sum())
            dr[n]=out
        draws[k]=dr; print('scored',k,{n:round(res[k][n],5) for n in names},getattr(F,'mix',''),flush=True)
    out=dict(res=res,names=names,ref=ref)
    # aggregates
    def agg(vals,kind,K):
        v=np.array(vals); return {s:float((weights(K,s)*v).sum()) for s in ('mean','lin','exp')}
    summ={}
    for n in names:
        summ[n]={'plain_K4':agg([res[f'plain{d}'][n] for d in (3,4,5,6)],'p',4),'plain_K3':agg([res[f'plain{d}'][n] for d in (4,5,6)],'p',3),
                 'sm_K3':agg([res[f'sm{d}'][n] for d in (4,5,6)],'s',3),'lastday':res['plain6'][n],'C23':res['C23'][n],'C25':res['C25'][n]}
    out['summary']=summ
    # bootstrap sd of deltas vs ref and of each aggregate
    def drawagg(n,keys,K,s): return sum(weights(K,s)[i]*draws[k][n] for i,k in enumerate(keys))
    dsd={}
    for n in names:
        if n==ref: continue
        e={}
        for lab,keys,K in (('plain_K4',['plain3','plain4','plain5','plain6'],4),('plain_K3',['plain4','plain5','plain6'],3),('sm_K3',['sm4','sm5','sm6'],3)):
            for s in ('mean','lin','exp'): x=drawagg(n,keys,K,s)-drawagg(ref,keys,K,s); e[f'{lab}_{s}']=float(x.std())
        for k in ('plain6','C23','C25','plain3','plain4','plain5','sm4','sm5','sm6'): e[k]=float((draws[k][n]-draws[k][ref]).std())
        dsd[n]=e
    out['delta_sd']=dsd; out['auc_sd']={k:{n:float(draws[k][n].std()) for n in names} for k in sets}
    out['mix']={k:getattr(sets[k][0],'mix',None) for k in sets}; out['n_slates']={k:int(sets[k][0].nev) for k in sets}
    json.dump(out,open(f'{R}/score_{a.tag}.json','w'),indent=1)
    print(json.dumps(out['summary'],indent=1)); print('delta_sd',json.dumps(dsd,indent=1))
if __name__=='__main__':
    ap=argparse.ArgumentParser(); sp=ap.add_subparsers(dest='cmd',required=True)
    sp.add_parser('days'); sp.add_parser('plan')
    f=sp.add_parser('fit'); f.add_argument('--variant',required=True); f.add_argument('--folds',default='3,4,5,6'); f.add_argument('--threads',type=int,default=4); f.add_argument('--seed',type=int,default=0); f.add_argument('--rounds',type=int,default=ROUNDS); f.add_argument('--lr',type=float,default=0.06)
    t=sp.add_parser('slate'); t.add_argument('--variant',required=True); t.add_argument('--folds',default='3,4,5,6'); t.add_argument('--threads',type=int,default=4); t.add_argument('--srounds',type=int,default=600); t.add_argument('--a',type=float,default=0.3); t.add_argument('--c',type=float,default=0.5)
    s=sp.add_parser('score'); s.add_argument('--names',required=True); s.add_argument('--ref'); s.add_argument('--pred-dir'); s.add_argument('--boot',type=int,default=200); s.add_argument('--tag',default='main')
    a=ap.parse_args(); {'days':cmd_days,'plan':cmd_plan,'fit':cmd_fit,'slate':cmd_slate,'score':cmd_score}[a.cmd](a)
