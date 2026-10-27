# Staleness test: feature builder. Reads lite/*.parquet (read-only), writes feat/*.parquet here.
import numpy as np, polars as pl, time, sys, gc
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
L='/workspace/huawei/lite'; O='/workspace/huawei/staleness_test/feat'
DAY=86400; J20=1737331200  # 2025-01-20 00:00 UTC
def J(d): return J20+(d-20)*DAY
SAMPLE=int(sys.argv[1]) if len(sys.argv)>1 else 100   # % of users kept as target rows
K=16
t0=time.time()
def log(*a): print(f'[{time.time()-t0:6.0f}s]',*a,flush=True)

ev=pl.read_parquet(f'{L}/train_events.parquet').sort('ts')
rows=pl.read_parquet(f'{L}/train_rows.parquet',columns=['eid','item','clicked'])
NE=ev.height
ev=ev.with_row_index('e')
rows=rows.join(ev.select('eid','e'),on='eid').sort(['e','item'])
assert rows.height==NE*10
uu,EU=np.unique(ev['user'].to_numpy(),return_inverse=True); EU=EU.astype(np.int32); NU=len(uu)
ETS=ev['ts'].to_numpy()
Hlen=ev['h_items'].list.len().to_numpy().astype(np.int64)
OFF=np.zeros(NE+1,np.int64); OFF[1:]=np.cumsum(Hlen)
HIraw=ev['h_items'].explode().drop_nulls().to_numpy(); HT=ev['h_ts'].explode().drop_nulls().to_numpy().astype(np.int64)
assert len(HIraw)==OFF[-1]
Rraw=rows['item'].to_numpy(); Y=rows['clicked'].to_numpy().astype(np.int8)
items=np.unique(np.concatenate([np.unique(Rraw),np.unique(HIraw)])); NI=len(items)
HI=np.searchsorted(items,HIraw).astype(np.int32); RI=np.searchsorted(items,Rraw).astype(np.int32)
del HIraw,Rraw,rows; ev=ev.select('e','eid','user','ts'); gc.collect()
# user sample by hashed user id (deterministic)
keep_user=((uu*2654435761)%1000003)%100 < SAMPLE
log('loaded',NE,'events',NU,'users',NI,'items','hist',OFF[-1])

def snapshot(X):
    """All stats from clicks/impressions strictly before X, using only train events with ts<X."""
    nX=np.searchsorted(ETS,X)
    hu=np.repeat(EU[:nX],Hlen[:nX]); hi=HI[:OFF[nX]]; ht=HT[:OFF[nX]]
    rr=np.nonzero(Y[:nX*10])[0]
    cu=np.concatenate([hu,EU[rr//10]]); ci=np.concatenate([hi,RI[rr]]); ct=np.concatenate([ht,ETS[rr//10]])
    tm=(ct-J(1))//60
    key=np.unique((cu.astype(np.int64)*NI+ci)*(1<<20)+tm)
    del cu,ci,ct,hu,hi,ht
    ui=key>>20; tm=key&((1<<20)-1); u=(ui//NI).astype(np.int32); i=(ui%NI).astype(np.int32)
    tsec=tm*60+J(1)
    S={'X':X}
    S['i_clk']=np.bincount(i,minlength=NI).astype(np.float32)
    S['i_clk24']=np.bincount(i[tsec>=X-DAY],minlength=NI).astype(np.float32)
    S['i_clk72']=np.bincount(i[tsec>=X-3*DAY],minlength=NI).astype(np.float32)
    S['u_clk']=np.bincount(u,minlength=NU).astype(np.float32)
    imp=np.bincount(RI[:nX*10],minlength=NI).astype(np.float32)
    rclk=np.bincount(RI[rr],minlength=NI).astype(np.float32)
    p0=rclk.sum()/max(imp.sum(),1)
    S['i_imp']=imp; S['i_ctr']=(rclk+20*p0)/(imp+20)
    ue=np.bincount(EU[:nX],minlength=NU).astype(np.float32); S['u_ev']=ue
    ucl=np.bincount(EU[rr//10],minlength=NU).astype(np.float32)
    S['u_ctr']=(ucl+5*0.629)/(ue+5)   # event-level click rate
    last=np.full(NU,-1,np.int64); np.maximum.at(last,EU[:nX],ETS[:nX]); S['u_lastev']=last
    # SVD taste vectors on binary user-item matrix
    ui2,cnt=np.unique(ui,return_counts=True); uu2=(ui2//NI).astype(np.int32); ii2=(ui2%NI).astype(np.int32)
    nu=np.bincount(uu2,minlength=NU).astype(np.float32)
    niv=np.bincount(ii2,minlength=NI).astype(np.float32)
    M=csr_matrix((1/np.sqrt((nu[uu2]+1)*(niv[ii2]+1)),(uu2,ii2)),shape=(NU,NI),dtype=np.float32)
    _,s,Vt=svds(M,k=K,random_state=0)
    E=(Vt.T*s).astype(np.float32); E/=np.linalg.norm(E,axis=1,keepdims=True)+1e-9
    S['emb']=E; del M,ui2,uu2,ii2
    # co-click: pairs of clicks within lag<=2 in each user's time-ordered click sequence
    o=np.lexsort((tm,u)); us=u[o]; is_=i[o]; keys=[]
    for lag in (1,2):
        m=(us[lag:]==us[:-lag])&(is_[lag:]!=is_[:-lag])
        a=is_[lag:][m].astype(np.int64); b=is_[:-lag][m].astype(np.int64)
        keys.append(np.minimum(a,b)*NI+np.maximum(a,b))
    kk=np.concatenate(keys); del keys; kk.sort()
    ck,cc=np.unique(kk,return_counts=True); S['cc_k']=ck; S['cc_v']=cc.astype(np.float32)
    log(f'snapshot {X} nX={nX} clicks={len(key)} pairs={len(ck)}')
    return S

def rank_in_event(v):  # v: (n,10) -> descending rank normalised 0..1
    r=np.argsort(np.argsort(-v,axis=1,kind='stable'),axis=1,kind='stable')
    return (r/9.0).astype(np.float32)

def features(E_idx,S):
    X=S['X']; n=len(E_idx); emb=S['emb']
    t=ETS[E_idx]; uidx=EU[E_idx]; L_=Hlen[E_idx]
    pos=np.arange(L_.sum())-np.repeat(np.cumsum(L_)-L_,L_)
    fl=np.repeat(OFF[E_idx],L_)+pos; g=np.repeat(np.arange(n),L_)
    hi=HI[fl]; ht=HT[fl]; tg=t[g]; Lg=L_[g]
    ri=(E_idx[:,None]*10+np.arange(10)).ravel(); cand=RI[ri]; rg=np.repeat(np.arange(n),10)
    F={}
    F['y']=Y[ri]; F['e']=np.repeat(E_idx,10); F['stale_h']=np.repeat(((t-X)/3600).astype(np.float32),10)
    # --- history aggregates
    ne=L_>0
    lastc=np.full(n,np.nan); 
    if ne.any():
        st=np.cumsum(L_)-L_
        lastc[ne]=ht[st[ne]+L_[ne]-1]   # histories sorted oldest->newest
    F['hist_len']=np.repeat(L_.astype(np.float32),10)
    warm=np.bincount(g,weights=(ht<J20),minlength=n)/np.maximum(L_,1)
    F['warm_share']=np.repeat(np.where(ne,warm,np.nan).astype(np.float32),10)
    # candidate-in-history: count, last occurrence age, position from end
    hk=g.astype(np.int64)*NI+hi; o=np.argsort(hk,kind='stable'); hks=hk[o]
    uk,st2,cn=np.unique(hks,return_index=True,return_counts=True)
    lastpos=np.maximum.reduceat(pos[o],st2) if len(hks) else np.array([],np.int64)
    lastts=np.maximum.reduceat(ht[o],st2) if len(hks) else np.array([],np.int64)
    rk=rg.astype(np.int64)*NI+cand; j=np.searchsorted(uk,rk); j=np.minimum(j,max(len(uk)-1,0))
    hit=(len(uk)>0)&(uk[j]==rk) if len(uk) else np.zeros(len(rk),bool)
    F['inh_cnt']=np.where(hit,cn[j],0).astype(np.float32)
    F['inh_age_h']=np.where(hit,(t[rg]-lastts[j])/3600,np.nan).astype(np.float32)
    F['inh_pos_end']=np.where(hit,L_[rg]-1-lastpos[j],np.nan).astype(np.float32)
    # popularity / item stats
    F['i_clk']=np.log1p(S['i_clk'][cand]); F['i_clk24']=np.log1p(S['i_clk24'][cand]); F['i_clk72']=np.log1p(S['i_clk72'][cand])
    F['i_imp']=np.log1p(S['i_imp'][cand]); F['i_ctr']=S['i_ctr'][cand]
    # user stats
    F['u_ev']=np.repeat(np.log1p(S['u_ev'][uidx]),10); F['u_ctr']=np.repeat(S['u_ctr'][uidx],10)
    F['u_clk']=np.repeat(np.log1p(S['u_clk'][uidx]),10)
    # taste similarity
    he=emb[hi]
    def uvec(w):
        V=np.stack([np.bincount(g,weights=he[:,k]*w,minlength=n) for k in range(K)],1).astype(np.float32)
        nr=np.linalg.norm(V,axis=1); nr[nr==0]=np.nan; return V/nr[:,None]
    ce=emb[cand]
    def cos(V): return (np.einsum('ij,ij->i',V[rg],ce)).astype(np.float32)
    F['sim']=cos(uvec(np.ones(len(hi),np.float32)))
    rec=(pos>=Lg-5).astype(np.float32)
    F['sim_rec5']=cos(uvec(rec)); F['sim_old']=cos(uvec(1-rec))
    F['sim_dec48']=cos(uvec(np.exp(-(tg-ht)/(48*3600)).astype(np.float32)))
    # co-click score vs history (chunked)
    ck,cv=S['cc_k'],S['cc_v']; ic=S['i_clk']
    ccs=np.zeros(n*10,np.float32); ccn=np.zeros(n*10,np.float32); ccm=np.zeros(n*10,np.float32)
    st=np.cumsum(L_)-L_; CH=20000
    for a in range(0,n,CH):
        b=min(n,a+CH); rr_=np.arange(a*10,b*10); ev_=rr_//10; Ln=L_[ev_]
        pr=np.repeat(rr_,Ln); pp=np.arange(Ln.sum())-np.repeat(np.cumsum(Ln)-Ln,Ln)
        h=hi[np.repeat(st[ev_],Ln)+pp].astype(np.int64); c=cand[pr].astype(np.int64)
        k=np.minimum(h,c)*NI+np.maximum(h,c); jj=np.minimum(np.searchsorted(ck,k),len(ck)-1)
        v=np.where((ck[jj]==k)&(h!=c),cv[jj],0).astype(np.float32)
        vn=v/np.sqrt((ic[h]+1)*(ic[c]+1))
        loc=pr-a*10
        ccs[rr_]=np.bincount(loc,weights=np.log1p(v),minlength=len(rr_))
        ccn[rr_]=np.bincount(loc,weights=vn,minlength=len(rr_))
        m=np.zeros(len(rr_),np.float32); nz=Ln>0
        if nz.any(): m[nz]=np.maximum.reduceat(vn,(np.cumsum(Ln)-Ln)[nz])
        ccm[rr_]=m
    F['cc_sum']=ccs; F['cc_nsum']=ccn; F['cc_nmax']=ccm
    # event-level context + within-event ranks
    for c in ['sim','cc_nsum','i_clk','inh_cnt','sim_dec48']:
        v=np.nan_to_num(F[c].reshape(n,10),nan=-9)
        F[f'r_{c}']=rank_in_event(v).ravel()
    s=np.nan_to_num(F['sim'].reshape(n,10),nan=-1)
    F['ev_sim_max']=np.repeat(s.max(1),10); F['ev_sim_mean']=np.repeat(s.mean(1),10)
    F['sim_gap']=(s-s.max(1,keepdims=True)).ravel()
    F['ev_inh_n']=np.repeat((F['inh_cnt'].reshape(n,10)>0).sum(1).astype(np.float32),10)
    F['ev_cc_max']=np.repeat(F['cc_nsum'].reshape(n,10).max(1),10)
    # ---- A: staleness / recency features
    F['A_since_click_h']=np.repeat(((t-lastc)/3600).astype(np.float32),10)
    F['A_click_to_cut_h']=np.repeat(((X-lastc)/3600).astype(np.float32),10)
    le=S['u_lastev'][uidx]; F['A_since_ev_h']=np.repeat(np.where(le>0,(t-le)/3600,np.nan).astype(np.float32),10)
    F['A_hist24']=np.repeat(np.bincount(g,weights=(tg-ht<=DAY),minlength=n).astype(np.float32),10)
    fts=np.full(n,np.nan); fts[ne]=ht[st[ne]]
    F['A_hist_span_h']=np.repeat(((t-fts)/3600).astype(np.float32),10)
    F['A_inh_x_stale']=np.where(F['inh_cnt']>0,F['stale_h'],np.nan).astype(np.float32)
    F['A_inh_age_rel']=(F['inh_age_h']/np.maximum(F['A_since_click_h'],1/60)).astype(np.float32)
    F['A_sim_diff']=(F['sim_rec5']-F['sim_old']).astype(np.float32)
    df=pl.DataFrame({k:v for k,v in F.items()})
    return df

def first_after(lo,hi_):
    m=(ETS>=lo)&(ETS<hi_)&keep_user[EU]
    idx=np.nonzero(m)[0]
    _,f=np.unique(EU[idx],return_index=True)   # ETS sorted -> first occurrence = first event
    return np.sort(idx[f])

END=J(27)
snaps={}
for X in [J(d) for d in (21,22,23,24,25)]:
    snaps[X]=snapshot(X); gc.collect()
    d=(X-J20)//DAY+20
    # targets that use snapshot X:
    jobs=[('plain',first_after(X,X+DAY)),('B',first_after(X,J(24))),('B',first_after(X,J(25))),('val',first_after(X,END))]
    names=[f'plain_P{d}', f'B_P{d}_C24', f'B_P{d}_C25', f'val_C{d}']
    for nm,(kind,idx) in zip(names,jobs):
        if kind=='B' and int(nm[-2:])<=d: continue
        if kind=='val' and d not in (24,25): continue
        if kind=='plain' and d>24: continue
        df=features(idx,snaps[X]); df.write_parquet(f'{O}/{nm}.parquet')
        log(nm,len(idx),'events', 'stale>=24h %.3f'%(df['stale_h']>=24).mean())
        del df; gc.collect()
    del snaps[X]; gc.collect()
log('done')
