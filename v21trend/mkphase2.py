#!/usr/bin/env python
"""v21trend/mkphase2.py -- Phase 2 features, leak-free (train-only graphs at cut C; query = row's own pre-ts history).
 feat_cs  : explicit cold-start flags (item no-impression in fit/24h/6h, user x item never shown, never clicked, slate counts)
 feat_co2 : two-hop item->nbr->item scores between candidate and own history (cos-S^2 and PPMI random-walk T^2) max/mean/last/mean3
 feat_cl  : item clusters (k=64,256 on train-only SVD embeddings of click matrix) + cluster->cluster transition / affinity features
Graph at cut C uses ONLY: fit-window clicks (ts<C) + hist of events with ts<C (no target-event histories, no eval rows)."""
import sys, glob, time, os, numpy as np, polars as pl
from scipy import sparse
from sklearn.utils.extmath import randomized_svd
from sklearn.cluster import MiniBatchKMeans
H='/workspace/huawei'; V=f'{H}/v21trend'; J20=1737331200; DAY=86400
CACHE=f'{H}/v9d/val/cache_mx'
CUT={'f1':1,'f2':2,'f3':3,'f4':4,'f5':5,'f6':6,'val23':3,'val25':5,'val26':6,'eval':7}
TR=pl.read_parquet(f'{H}/lite/train_rows.parquet'); TE=pl.read_parquet(f'{H}/lite/train_events.parquet')
ER=pl.read_parquet(f'{H}/lite/evaluation_rows.parquet'); EE=pl.read_parquet(f'{H}/lite/evaluation_events.parquet')
ITEMS=np.unique(np.concatenate([TR['item'].unique().to_numpy(),ER['item'].unique().to_numpy(),TE['h_items'].explode().drop_nulls().unique().to_numpy(),EE['h_items'].explode().drop_nulls().unique().to_numpy()]))
NI=len(ITEMS); USERS=np.unique(np.concatenate([TE['user'].unique().to_numpy(),EE['user'].unique().to_numpy()])); NU=len(USERS)
def ix(s): return pl.Series(np.searchsorted(ITEMS,s.to_numpy()).astype(np.int32))
ROWS=TR.join(TE.select('eid','user','ts'),on='eid').with_columns(ix(pl.col('item')).alias('ii')) if False else None
ROWS=TR.join(TE.select('eid','user','ts'),on='eid'); ROWS=ROWS.with_columns(pl.Series('ii',np.searchsorted(ITEMS,ROWS['item'].to_numpy()).astype(np.int32)))
def rowids(n):
    if n=='eval': return ER['rowid']
    g=[x for x in glob.glob(f'{CACHE}/{n}_*.parquet') if not x.endswith('.json')]; return pl.read_parquet(g[0],columns=['rowid'])['rowid']
def topk_rows(M,K):
    """keep top-K entries per row of csr M"""
    M=M.tocsr(); ind=[];dat=[];ptr=[0]
    for r in range(M.shape[0]):
        a,b=M.indptr[r],M.indptr[r+1]; d=M.data[a:b]; c=M.indices[a:b]
        if b-a>K: o=np.argpartition(-d,K-1)[:K]; d=d[o]; c=c[o]
        ind.append(c);dat.append(d);ptr.append(ptr[-1]+len(d))
    return sparse.csr_matrix((np.concatenate(dat).astype(np.float32),np.concatenate(ind),np.array(ptr)),shape=M.shape)
def graph(C):
    P_ev=TE.filter(pl.col('ts')<C); P=ROWS.filter(pl.col('ts')<C)
    win=P.filter(pl.col('clicked')==1).select('user','ii',pl.col('ts'))
    hist=(P_ev.select('user',pl.col('h_items').alias('hi'),pl.col('h_ts').alias('ht')).explode(['hi','ht']).drop_nulls('hi').filter(pl.col('ht')<C)
          .select('user',pl.Series('ii',np.zeros(0,np.int32)) if False else pl.col('hi').alias('item'),pl.col('ht').alias('ts')))
    hist=hist.with_columns(pl.Series('ii',np.searchsorted(ITEMS,hist['item'].to_numpy()).astype(np.int32))).select('user','ii','ts')
    CL=pl.concat([win,hist]).unique().sort('user','ts','ii')
    u=np.searchsorted(USERS,CL['user'].to_numpy()); i=CL['ii'].to_numpy()
    B=sparse.csr_matrix((np.ones(len(u),np.float32),(u,i)),shape=(NU,NI)); B.data[:]=1; B.sum_duplicates(); B.data[:]=1
    return P,P_ev,CL,B,u,i
def mats(B,K=64):
    pop=np.asarray(B.sum(0)).ravel().astype(np.float64); Co=(B.T@B).tocsr(); Co.setdiag(0); Co.eliminate_zeros()
    # cosine
    d=sparse.diags(1/np.sqrt(pop+1e-9)); S=(d@Co@d).tocsr().astype(np.float32); S=topk_rows(S,K)
    # PPMI -> row-stochastic walk
    Co=Co.tocoo(); N=B.shape[0]; pm=np.log(Co.data*N/(pop[Co.row]*pop[Co.col]+1e-9)); keep=pm>0
    Pp=sparse.csr_matrix((pm[keep].astype(np.float32),(Co.row[keep],Co.col[keep])),shape=Co.shape); Pp=topk_rows(Pp,K)
    rs=np.asarray(Pp.sum(1)).ravel(); T=sparse.diags(1/(rs+1e-9))@Pp; T=T.tocsr().astype(np.float32)
    return S,T
def pairdot(A,B_,j,i,chunk=400000):
    """sum_m A[j,m]*B_[i,m] for pair arrays; A,B_ csr (row-gather, elementwise multiply)"""
    out=np.zeros(len(j),np.float32)
    for s in range(0,len(j),chunk):
        a=A[j[s:s+chunk]]; b=B_[i[s:s+chunk]]; out[s:s+chunk]=np.asarray(a.multiply(b).sum(1)).ravel()
    return out
def tables_for(C): return [n for n,c in CUT.items() if J20+c*DAY==C]
def target(n):
    rid=rowids(n).to_frame()
    if n=='eval': T=rid.join(ER,on='rowid').join(EE,on='eid')
    else: T=rid.join(TR.select('rowid','eid','item'),on='rowid').join(TE,on='eid')
    T=T.with_columns(pl.Series('ii',np.searchsorted(ITEMS,T['item'].to_numpy()).astype(np.int32))).with_row_index('r')
    return T
def hist_pairs(T,maxpos=10):
    E=T.select('eid','h_items','h_ts','ts').unique('eid')
    L=(E.select('eid','ts',pl.col('h_items').alias('hi'),pl.col('h_ts').alias('ht')).explode(['hi','ht']).drop_nulls('hi').filter(pl.col('ht')<pl.col('ts'))
        .sort('eid','ht').with_columns((pl.len().over('eid')-pl.col('ht').cum_count().over('eid')+1).alias('pos')))
    L=L.with_columns(pl.Series('j',np.searchsorted(ITEMS,L['hi'].to_numpy()).astype(np.int32))).select('eid','pos','j').filter(pl.col('pos')<=20)
    return L
def build_cut(C):
    t0=time.time(); P,P_ev,CL,B,cu,ci=graph(C); nd=int(round((C-J20)/DAY)); g=float(P['clicked'].mean())
    S,T_=mats(B); print(' graph',C,f'{time.time()-t0:.0f}s nnz S {S.nnz} T {T_.nnz}',flush=True)
    TT=T_.T.tocsr()                    # column access: TT[i,m]=T[m,i]
    # item fit-window stats (for flags / cluster ctr)
    r1=pl.col('ts')>=C-DAY; r6=pl.col('ts')>=C-6*3600
    it=P.group_by('ii').agg(pl.len().alias('imp'),r1.sum().alias('imp24'),r6.sum().alias('imp6'),pl.col('clicked').sum().alias('clk'))
    imp=np.zeros(NI,np.int64);imp24=np.zeros(NI,np.int64);imp6=np.zeros(NI,np.int64);clk=np.zeros(NI,np.int64)
    a=it['ii'].to_numpy(); imp[a]=it['imp'].to_numpy(); imp24[a]=it['imp24'].to_numpy(); imp6[a]=it['imp6'].to_numpy(); clk[a]=it['clk'].to_numpy()
    # user x item shown / clicked sets (train-only: P rows) as hashed keys
    ukey=(np.searchsorted(USERS,P['user'].to_numpy()).astype(np.int64))*NI+P['ii'].to_numpy().astype(np.int64); shown=np.unique(ukey)
    cn=np.unique(cu.astype(np.int64)*NI+ci.astype(np.int64))
    pe_users=np.unique(np.searchsorted(USERS,P_ev['user'].to_numpy()))
    # item embeddings + clusters
    cnt=B.copy(); df=np.asarray(cnt.sum(0)).ravel(); idf=np.log((NU+1)/(df+1)).astype(np.float32)
    Xc=cnt@sparse.diags(idf); rn=np.sqrt(np.asarray(Xc.multiply(Xc).sum(1)).ravel())+1e-6; Xc=sparse.diags(1/rn)@Xc
    U,Sg,Vt=randomized_svd(Xc,32,random_state=0,n_iter=5); Vm=(Vt.T*np.sqrt(Sg)).astype(np.float32); nrm=np.linalg.norm(Vm,axis=1,keepdims=True); Vn=Vm/(nrm+1e-9)
    has=(df>0)
    CLU={}
    for K in (64,256):
        km=MiniBatchKMeans(K,random_state=0,n_init=3,batch_size=4096,max_iter=60).fit(Vn[has]); c=np.full(NI,-1,np.int32); c[has]=km.predict(Vn[has]); CLU[K]=c
    # cluster transitions (lag 1 consecutive clicks per user, train-only click log), smoothed
    cu_s=CL['user'].to_numpy(); cj=CL['ii'].to_numpy(); same=cu_s[1:]==cu_s[:-1]; pj=cj[:-1][same]; nj=cj[1:][same]
    CT={}
    for K in (64,256):
        c=CLU[K]; a=c[pj]; b=c[nj]; k=(a>=0)&(b>=0); M=np.zeros((K,K),np.float64); np.add.at(M,(a[k],b[k]),1)
        marg=(np.bincount(c[cj][c[cj]>=0],minlength=K)+1.0); marg=marg/marg.sum(); al=20.0
        Pm=(M+al*marg[None,:])/(M.sum(1,keepdims=True)+al); CT[K]=(Pm.astype(np.float32),(Pm/marg[None,:]).astype(np.float32),marg.astype(np.float32))
        # cluster impression CTR
    clctr={}
    for K in (64,256):
        c=CLU[K]; ok=c>=0; ci_=np.where(ok,c,0); si=np.bincount(ci_[ok],weights=imp[ok],minlength=K); sc=np.bincount(ci_[ok],weights=clk[ok],minlength=K); clctr[K]=((sc+200*g)/(si+200)).astype(np.float32)
    for n in tables_for(C):
        t1=time.time(); T=target(n); Rr=T.select('r','rowid','eid','ii','user').sort('r'); N=Rr.height
        ii=Rr['ii'].to_numpy(); uu=np.searchsorted(USERS,Rr['user'].to_numpy()).astype(np.int64)
        # ---- feat_cs ----
        cs={'cs_it_noimp':(imp[ii]==0),'cs_it_noimp24':(imp24[ii]==0),'cs_it_noimp6':(imp6[ii]==0),'cs_it_lowimp20':(imp[ii]<20),
            'cs_ui_never_shown':~np.isin(uu*NI+ii,shown),'cs_ui_never_clk':~np.isin(uu*NI+ii,cn),'cs_user_cold':~np.isin(uu,pe_users)}
        D=pl.DataFrame({k:v.astype(np.int8) for k,v in cs.items()}).with_columns(Rr['eid'].alias('eid'))
        for k in ('cs_it_noimp6','cs_it_noimp24','cs_ui_never_shown'): D=D.with_columns(pl.col(k).sum().over('eid').alias(k+'_slate'))
        D=D.with_columns(((pl.col('cs_user_cold')==1)|(pl.col('cs_it_noimp24')==1)).cast(pl.Int8).alias('cs_cold_any')).drop('eid')
        D.insert_column(0,Rr['rowid']); D.write_parquet(f'{V}/feat_cs/{n}.parquet')
        # ---- history pairs ----
        L=hist_pairs(T); P_=Rr.select('r','eid','ii').join(L,on='eid',how='inner')   # r,eid,ii,pos,j
        P_=P_.filter(pl.col('pos')<=10)
        j=P_['j'].to_numpy(); i2=P_['ii'].to_numpy(); pos=P_['pos'].to_numpy(); r=P_['r'].to_numpy()
        key=j.astype(np.int64)*NI+i2; uk,inv=np.unique(key,return_inverse=True); uj=(uk//NI).astype(np.int64); ui=(uk%NI).astype(np.int64)
        s2=pairdot(S,S,uj,ui)[inv]; t2=pairdot(T_,TT,uj,ui)[inv]
        s2[j==i2]=0; t2[j==i2]=0
        df_=pl.DataFrame({'r':r,'pos':pos,'s2':s2,'t2':t2})
        agg=df_.group_by('r').agg(pl.col('s2').max().alias('co2_cos_max10'),pl.col('s2').sum().alias('co2_cos_sum10'),pl.col('s2').filter(pl.col('pos')==1).first().alias('co2_cos_last'),pl.col('s2').filter(pl.col('pos')<=3).mean().alias('co2_cos_mean3'),
            pl.col('t2').max().alias('co2_rw_max10'),pl.col('t2').sum().alias('co2_rw_sum10'),pl.col('t2').filter(pl.col('pos')==1).first().alias('co2_rw_last'),pl.col('t2').filter(pl.col('pos')<=3).mean().alias('co2_rw_mean3'))
        Q=Rr.select('r','rowid','eid').join(agg,on='r',how='left').sort('r'); cc=[c for c in Q.columns if c.startswith('co2')]
        Q=Q.with_columns([pl.col(c).fill_null(0) for c in cc])
        for c in ('co2_cos_max10','co2_rw_max10','co2_rw_sum10'):
            x=pl.col(c).cast(pl.Float64); Q=Q.with_columns(x.rank('average').over('eid').alias(c+'_rk'),((x-x.mean().over('eid'))/(x.std().over('eid')+1e-12)).alias(c+'_z'))
        cc=[c for c in Q.columns if c.startswith('co2')]
        Q.select(['rowid']+cc).with_columns([pl.col(c).cast(pl.Float32) for c in cc]).write_parquet(f'{V}/feat_co2/{n}.parquet')
        # ---- clusters ----
        outc={}
        hj=P_.select('r','pos','j')
        for K in (64,256):
            c=CLU[K]; Pm,Lf,mg=CT[K]; cand=c[ii]; ch=c[hj['j'].to_numpy()]; rr=hj['r'].to_numpy(); ps=hj['pos'].to_numpy(); cd=cand[rr]
            ok=(ch>=0)&(cd>=0); pt=np.where(ok,Pm[np.where(ok,ch,0),np.where(ok,cd,0)],np.nan); lf=np.where(ok,Lf[np.where(ok,ch,0),np.where(ok,cd,0)],np.nan)
            sm=(ch==cd)&ok
            d=pl.DataFrame({'r':rr,'pos':ps,'pt':pt,'lf':lf,'sm':sm.astype(np.float32)})
            ag=d.group_by('r').agg(pl.col('pt').filter(pl.col('pos')==1).first().alias(f'cl{K}_tr_last'),pl.col('lf').filter(pl.col('pos')==1).first().alias(f'cl{K}_lift_last'),
               pl.col('pt').filter(pl.col('pos')<=3).mean().alias(f'cl{K}_tr_mean3'),pl.col('lf').filter(pl.col('pos')<=3).mean().alias(f'cl{K}_lift_mean3'),
               pl.col('lf').mean().alias(f'cl{K}_lift_mean10'),pl.col('lf').max().alias(f'cl{K}_lift_max10'),pl.col('sm').filter(pl.col('pos')==1).first().alias(f'cl{K}_same_last'),
               pl.col('sm').filter(pl.col('pos')<=5).mean().alias(f'cl{K}_frac5'),pl.col('sm').mean().alias(f'cl{K}_frac10'))
            Z=Rr.select('r','eid').join(ag,on='r',how='left').sort('r').with_columns(pl.Series(f'cl{K}_ctr',np.where(cand>=0,clctr[K][np.where(cand>=0,cand,0)],np.nan)),pl.Series(f'cl{K}_marg',np.where(cand>=0,mg[np.where(cand>=0,cand,0)],np.nan)))
            for cname in (f'cl{K}_lift_mean10',f'cl{K}_lift_last'):
                x=pl.col(cname).cast(pl.Float64); Z=Z.with_columns(x.rank('average').over('eid').alias(cname+'_rk'),((x-x.mean().over('eid'))/(x.std().over('eid')+1e-9)).alias(cname+'_z'))
            for cname in Z.columns:
                if cname.startswith('cl'): outc[cname]=Z[cname].to_numpy()
        Fc=pl.DataFrame({k:v.astype(np.float32) for k,v in outc.items()}); Fc.insert_column(0,Rr['rowid']); Fc.write_parquet(f'{V}/feat_cl/{n}.parquet')
        print(' ',n,'rows',N,'pairs',len(j),'uniq',len(uk),'co2',len(cc),'cl',Fc.width-1,f'{time.time()-t1:.0f}s',flush=True)
    print('cut',C,f'total {time.time()-t0:.0f}s',flush=True)
if __name__=='__main__':
    cuts=[J20+c*DAY for c in (sys.argv[1:] and [int(x) for x in sys.argv[1:]] or [1,2,3,4,5,6,7])]
    for d in ('feat_cs','feat_co2','feat_cl'): os.makedirs(f'{V}/{d}',exist_ok=True)
    for C in cuts: build_cut(C)
