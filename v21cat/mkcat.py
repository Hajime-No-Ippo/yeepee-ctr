"""mkcat.py DAYS... : build ordered/time-respecting target-encoding feature tables for cutoff day k (X=J20+k*DAY, only train events ts<X).
Tables per cutoff: 1:f1 2:f2 3:f3,val23 4:f4 5:f5,val25 6:f6,val26 7:eval.  Output cache/{T}.npy float32 (N,F) + cache/names.json. Row order == feat_v3/{T}.parquet (eval: seqfeat A_eval rowid order checked later)."""
import sys,os,time,json; sys.dont_write_bytecode=True
sys.path.insert(0,'/workspace/huawei/dw_signals'); os.chdir('/workspace/huawei/v21cat')
from common import *
from sklearn.cluster import MiniBatchKMeans
import scipy.sparse as sp
TAB={1:['f1'],2:['f2'],3:['f3','val23'],4:['f4'],5:['f5','val25'],6:['f6','val26'],7:['eval']}
D=Data(); NT=D.NE; n_orig=D.NU
# ---- extend with eval events (copy of mkdw.extend_eval)
def extend_eval(D):
    ee=pl.read_parquet('/workspace/huawei/lite/evaluation_events.parquet').sort('ts','eid'); er=pl.read_parquet('/workspace/huawei/lite/evaluation_rows.parquet')
    n=ee.height; eid2e={x:i for i,x in enumerate(ee['eid'].to_list())}
    er=er.with_columns(pl.col('eid').replace_strict(eid2e).alias('_e')).sort('_e','rowid')
    pos=np.searchsorted(D.uu,ee['user'].to_numpy()); pos=np.minimum(pos,D.NU-1); seen=D.uu[pos]==ee['user'].to_numpy()
    eu=np.where(seen,pos,D.NU).astype(np.int32)
    ri=np.searchsorted(D.items,er['item'].to_numpy()).astype(np.int32)
    hi=ee['h_items'].explode().drop_nulls().to_numpy(); ht=ee['h_ts'].explode().drop_nulls().to_numpy().astype(np.int64); hl=ee['h_items'].list.len().to_numpy()
    hp=np.minimum(np.searchsorted(D.items,hi),D.NI-1); ok=D.items[hp]==hi
    hl=hl.astype(np.int64); cs=np.concatenate([[0],np.cumsum(ok.astype(np.int64))]); ce=np.concatenate([[0],np.cumsum(hl)]); hl2=cs[ce[1:]]-cs[ce[:-1]]
    D0=D.NE; D.ETS=np.concatenate([D.ETS,ee['ts'].to_numpy()]); D.EU=np.concatenate([D.EU,eu]); D.Hlen=np.concatenate([D.Hlen,hl2.astype(np.int64)])
    base=D.OFF[-1]; D.OFF=np.concatenate([D.OFF[:-1],base+np.concatenate([[0],np.cumsum(hl2)])]).astype(np.int64)
    D.HI=np.concatenate([D.HI,hp[ok].astype(np.int32)]); D.HT=np.concatenate([D.HT,ht[ok]])
    D.RI=np.concatenate([D.RI,ri.reshape(n,10)]); D.Y=np.concatenate([D.Y,np.zeros((n,10),np.int8)]); D.NE=D0+n; D.NU=D.NU+1
    e=np.repeat(np.arange(D0,D0+n),10)
    return dict(rowid=er['rowid'].to_numpy(),e=e,item=ri)
EV=extend_eval(D)
def rows(name):
    if name=='eval': return EV['rowid'],EV['e']
    v=pl.read_parquet(f'/workspace/huawei/v9/feat_v3/{name}.parquet',columns=['rowid'])
    m=v.join(D.rows_rowid,on='rowid',how='left',maintain_order='left'); assert m['e'].null_count()==0
    return v['rowid'].to_numpy(),m['e'].to_numpy().astype(np.int64)
ENT=D.NE; HL=D.Hlen
Hm=sp.csr_matrix((np.ones(len(D.HI),np.float32),D.HI,D.OFF),shape=(ENT,D.NI))
hour_ev=(((D.ETS-J20)//3600)%24).astype(np.int64); dt_ev=((((D.ETS-J20)//DAY)%7)>=5).astype(np.int64)
class Agg:
    def __init__(s,keys,y):
        z=np.sort(keys.astype(np.int64)*2+y); kk=z>>1; yy=(z&1).astype(np.int64)
        st=np.concatenate([[0],np.flatnonzero(np.diff(kk))+1]); s.u=kk[st]; s.n=np.diff(np.concatenate([st,[len(z)]])).astype(np.float32); s.c=np.add.reduceat(yy,st).astype(np.float32)
    def q(s,k):
        i=np.searchsorted(s.u,k); i[i>=len(s.u)]=0; h=s.u[i]==k
        return np.where(h,s.n[i],0).astype(np.float32),np.where(h,s.c[i],0).astype(np.float32)
def ctr(n,c,prior,a): return ((c+a*prior)/(n+a)).astype(np.float32)
def build(day):
    t0=time.time(); X=J20+day*DAY; nX=int(np.searchsorted(D.ETS,X)); assert nX<=NT
    S=D.snap(X); V,Uu=click_svd(S,D,32)   # item vecs L2-normed
    cents={}
    for K in (48,256):
        km=MiniBatchKMeans(K,random_state=0,batch_size=4096,n_init=1,max_iter=60).fit(V); cents[K]=km
    cl={K:cents[K].predict(V).astype(np.int64) for K in cents}
    # user hist cluster per event (all events)
    UV=Hm@V; nr=np.linalg.norm(UV,axis=1); has=(HL>0)&(nr>1e-6); UVn=UV/(nr[:,None]+1e-9)
    ucl={}
    for K in cents:
        Cn=cents[K].cluster_centers_; Cn=Cn/(np.linalg.norm(Cn,axis=1,keepdims=True)+1e-9)
        a=np.argmax(UVn@Cn.T,1).astype(np.int64); ucl[K]=np.where(has,a,K)
    print(f'day{day} svd+km {time.time()-t0:.0f}s',flush=True)
    # ---- train rows
    e_t=np.repeat(np.arange(nX),10); it_t=D.RI[:nX].reshape(-1).astype(np.int64); y_t=D.Y[:nX].reshape(-1).astype(np.int64)
    u_t=D.EU[:nX][e_t].astype(np.int64); h_t=hour_ev[:nX][e_t]; dt_t=dt_ev[:nX][e_t]
    p0=y_t.mean(); NI=D.NI
    A={}
    A['i']=Agg(it_t,y_t); A['u']=Agg(u_t,y_t); A['i_tb']=Agg(it_t*4+h_t//6,y_t); A['i_h3']=Agg(it_t*8+h_t//3,y_t); A['i_dt']=Agg(it_t*2+dt_t,y_t)
    A['ui']=Agg(u_t*NI+it_t,y_t); A['u_tb']=Agg(u_t*4+h_t//6,y_t); A['u_dt']=Agg(u_t*2+dt_t,y_t)
    for K in (48,256):
        icl_t=cl[K][it_t]; uc_t=ucl[K][:nX][e_t]
        A[f'uic{K}']=Agg(u_t*(K+1)+icl_t,y_t); A[f'ucli{K}']=Agg(uc_t*NI+it_t,y_t); A[f'uclicl{K}']=Agg(uc_t*(K+1)+icl_t,y_t)
        A[f'iclh{K}']=Agg(icl_t*24+h_t,y_t)
    last_t=np.where(HL[:nX]>0,D.HI[np.maximum(D.OFF[1:nX+1]-1,0)],-1).astype(np.int64)  # newest hist item per event
    ok=np.repeat(last_t>=0,10); A['p0']=Agg(np.repeat(last_t,10)[ok]*NI+it_t[ok],y_t[ok])
    ks=[];ys=[]
    for j in range(5):
        okj=HL[:nX]>j; hj=np.where(okj,D.HI[np.maximum(D.OFF[1:nX+1]-1-j,0)],0).astype(np.int64); m=np.repeat(okj,10)
        ks.append(np.repeat(hj,10)[m]*NI+it_t[m]); ys.append(y_t[m])
    A['p5']=Agg(np.concatenate(ks),np.concatenate(ys)); del ks,ys
    # cluster of last hist item x cand
    ok=np.repeat(last_t>=0,10); A['pcl']=Agg(cl[48][np.repeat(last_t,10)[ok]]*NI+it_t[ok],y_t[ok])
    print(f'day{day} aggs {time.time()-t0:.0f}s',flush=True)
    iA=A['i']; ni_g,ci_g=iA.n,iA.c
    for T in TAB[day]:
        rid,e=rows(T); N=len(e); it=D.RI[e].reshape(-1) if False else None
        cand=D.RI[e[::10]].reshape(-1).astype(np.int64) if False else None
        ev=e[::10]; assert (np.repeat(ev,10)==e).all(); cand=D.RI[ev].reshape(-1).astype(np.int64); rep=np.repeat(np.arange(len(ev)),10)
        u=D.EU[ev][rep].astype(np.int64); h=hour_ev[ev][rep]; dt=dt_ev[ev][rep]; F={}
        ni,ci=A['i'].q(cand); i_ctr=ctr(ni,ci,p0,20); F['i_n']=ni; F['i_ctr']=i_ctr
        nu,cu=A['u'].q(np.where(u<n_orig,u,-1)); u_ctr=ctr(nu,cu,p0,20); F['u_n']=nu; F['u_ctr']=u_ctr
        def add(nm,agg,key,prior,a):
            n,c=agg.q(key); F[nm+'_n']=n; F[nm+'_ctr']=ctr(n,c,prior,a); return n,c
        add('i_tb',A['i_tb'],cand*4+h//6,i_ctr,10); add('i_h3',A['i_h3'],cand*8+h//3,i_ctr,10); add('i_dt',A['i_dt'],cand*2+dt,i_ctr,10)
        n,c=add('ui',A['ui'],np.where(u<n_orig,u,-1)*NI+cand,u_ctr,3); F['ui_c']=c
        add('u_tb',A['u_tb'],u*4+h//6,u_ctr,10); add('u_dt',A['u_dt'],u*2+dt,u_ctr,10)
        for K in (48,256):
            icl=cl[K][cand]; uc=ucl[K][ev][rep]
            add(f'uic{K}',A[f'uic{K}'],np.where(u<n_orig,u,-1)*(K+1)+icl,u_ctr,5); add(f'ucli{K}',A[f'ucli{K}'],uc*NI+cand,i_ctr,5)
            add(f'uclicl{K}',A[f'uclicl{K}'],uc*(K+1)+icl,i_ctr,10); add(f'iclh{K}',A[f'iclh{K}'],icl*24+h,i_ctr,10)
            F[f'ucl{K}']=uc.astype(np.float32); F[f'icl{K}']=icl.astype(np.float32)
        hl=HL[ev]; last=np.where(hl>0,D.HI[np.maximum(D.OFF[ev+1]-1,0)],-1).astype(np.int64)[rep]
        n,c=add('p0',A['p0'],np.where(last>=0,last,-1)*NI+cand,i_ctr,5); F['p0_n']=np.where(last>=0,n,np.nan)
        pc=[];pn=[]
        for j in range(5):
            oke=hl>j; okj=oke[rep]; hj=np.where(oke,D.HI[np.maximum(D.OFF[ev+1]-1-j,0)],0).astype(np.int64)[rep]
            n,c=A['p5'].q(hj*NI+cand); r=np.where(okj,ctr(n,c,i_ctr,5),np.nan); pc.append(r); pn.append(np.where(okj,n,np.nan))
        pc=np.stack(pc,1); pn=np.stack(pn,1)
        with np.errstate(all='ignore'):
            lg=np.log(pc/(1-pc))-np.log(i_ctr/(1-i_ctr))[:,None]
            F['p5_mean']=np.nanmean(pc,1); F['p5_max']=np.nanmax(pc,1); F['p5_min']=np.nanmin(pc,1); F['p5_lift']=np.nansum(lg,1); F['p5_nsum']=np.nansum(pn,1)
            F['p5_wmean']=np.nansum(pc*pn,1)/np.maximum(np.nansum(pn,1),1)
        F['p5_c0']=pc[:,0]; F['p5_c1']=pc[:,1]; F['p5_c2']=pc[:,2]; F['p5_n0']=pn[:,0]
        n,c=add('pcl',A['pcl'],np.where(last>=0,cl[48][np.maximum(last,0)],-1)*NI+cand,i_ctr,10)
        # in-history of cand
        cnt=np.zeros(len(cand),np.float32); lastj=np.full(len(cand),np.nan,np.float32); gap=np.full(len(cand),np.nan,np.float32)
        for j in range(20):
            oke=hl>j; okj=oke[rep]; idx=np.where(oke,D.OFF[ev+1]-1-j,0)[rep]; m=okj&(D.HI[idx]==cand); cnt+=m
            new=m&np.isnan(lastj); lastj[new]=j; gap[new]=np.log1p((D.ETS[ev][rep]-D.HT[idx]).clip(0)/3600)[new]
        F['inh_cnt']=cnt; F['inh_pos']=lastj; F['inh_gap']=gap
        F['hlen']=hl[rep].astype(np.float32); st=np.where(hl>0,D.ETS[ev]-D.HT[np.maximum(D.OFF[ev+1]-1,0)],np.nan); F['stale_h']=np.log1p(st/3600).astype(np.float32)[rep]
        F['hour']=h.astype(np.float32); F['dt']=dt.astype(np.float32); F['item_id']=cand.astype(np.float32)
        names=list(F); M=np.stack([np.asarray(F[k],np.float32) for k in names],1)
        np.save(f'cache/{T}.npy',M); np.save(f'cache/{T}_rowid.npy',rid); json.dump(names,open('cache/names.json','w'))
        print(f'day{day} {T} {M.shape} {time.time()-t0:.0f}s',flush=True)
if __name__=='__main__':
    for d in sys.argv[1:]: build(int(d))
