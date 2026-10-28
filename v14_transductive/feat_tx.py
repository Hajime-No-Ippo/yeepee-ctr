"""Priority 5 (TRANSDUCTIVE, label-free): statistics over the unlabelled TARGET rows (validation first-event rows when validating; evaluation rows for v14).
Uses only: target slates (10 candidate items), target event ts, target histories (items+ts), and train-side exposure/CTR strictly before C (labels of ts<C only).
No target-window labels are used anywhere."""
import sys; sys.path.insert(0,'/workspace/huawei/dw_lb')
from lib import *
from scipy.sparse import csr_matrix
import dwcommon as D
FAMS=['f1','f2','f3','f4']
def train_ref(d,C):
    """train-side reference from events ts<C (labels of ts<C only, allowed)"""
    nX=int(np.searchsorted(d.ETS,C)); nL=int(np.searchsorted(d.ETS,C-DAY))
    ex_all=np.bincount(d.RI[:nX].ravel(),minlength=d.NI).astype(np.float32); cl_all=np.bincount(d.RI[:nX].ravel(),weights=d.Y[:nX].ravel(),minlength=d.NI).astype(np.float32)
    ex_l=np.bincount(d.RI[nL:nX].ravel(),minlength=d.NI).astype(np.float32); cl_l=np.bincount(d.RI[nL:nX].ravel(),weights=d.Y[nL:nX].ravel(),minlength=d.NI).astype(np.float32)
    return dict(nX=nX,nL=nX-nL,ex_all=ex_all,cl_all=cl_all,ex_l=ex_l,cl_l=cl_l)
def target_from_data(d,e):
    """target events e (indices into d) -> slates, ts, histories (as unlabeled inputs)"""
    S=d.RI[e]; ts=d.ETS[e]; hi=[d.HI[d.OFF[x]:d.OFF[x+1]] for x in e]; ht=[d.HT[d.OFF[x]:d.OFF[x+1]] for x in e]
    return S,ts,hi,ht
def zrk(x):
    m=x.reshape(-1,10); mu=np.nanmean(m,1,keepdims=True); sd=np.nanstd(m,1,keepdims=True)+1e-6
    return ((m-mu)/sd).ravel(),(m-mu).ravel(),m.argsort(1).argsort(1).astype(np.float32).ravel()
def build_tx(S,ts,hi,ht,C,ref,NI,fams=FAMS):
    n=len(S); col={}; fl=S.ravel(); rep=lambda a:np.repeat(np.asarray(a,np.float32),10)
    cnt=np.bincount(fl,minlength=NI).astype(np.float32)   # exposure in target slates (includes own slot)
    # ---- family 1: item exposure in target slates
    if 'f1' in fams:
        c1=cnt[fl]-1; col['f1_cnt']=c1/n*1e4
        o=np.argsort(ts,kind='stable'); tss=ts[o]
        # time-windowed exposure: key=(item,ts) of every target slot
        its=np.repeat(S[o].ravel(),1); kt=np.repeat(tss,10); key=np.sort(its.astype(np.int64)*(1<<33)+kt)
        base=fl.astype(np.int64)*(1<<33); tr=np.repeat(ts,10).astype(np.int64)
        for nm,w in (('h1',3600),('h6',6*3600),('h24',24*3600)):
            lo=np.searchsorted(key,base+tr-w,'left'); hi_=np.searchsorted(key,base+tr+w,'right'); cw=(hi_-lo-1).astype(np.float32)
            nev=(np.searchsorted(tss,ts+w,'right')-np.searchsorted(tss,ts-w,'left')).astype(np.float32)   # #target events in the window
            col['f1_cnt_'+nm]=cw/np.repeat(nev,10)*1e3   # per-event normalised local exposure share
            if nm=='h6': col['f1_cnt_h6_raw']=cw
        expl=ref['ex_l'][fl]/max(ref['nL'],1)*1e3; col['f1_late_exp']=expl; col['f1_late_ctr']=(ref['cl_l'][fl]+0.2)/(ref['ex_l'][fl]+2)
        col['f1_ratio_late']=(c1/n+1e-4)/((ref['ex_l'][fl]+0.5)/max(ref['nL'],1)); col['f1_ratio_all']=(c1/n+1e-4)/((ref['ex_all'][fl]+0.5)/max(ref['nX'],1))
        col['f1_exp_x_ctr']=col['f1_cnt']*col['f1_late_ctr']
        for k in ('f1_cnt','f1_cnt_h6','f1_ratio_late'):
            z,dm,rk=zrk(col[k]); col[k+'_z']=z; col[k+'_dm']=dm; col[k+'_rk']=rk
        col['f1_slate_mean']=np.repeat(col['f1_cnt'].reshape(-1,10).mean(1),10)
    # ---- family 3: popularity in the histories of target users (items clicked; incl. post-cutoff clicks visible in histories)
    hlen=np.array([len(x) for x in hi]); hI=np.concatenate(hi).astype(np.int64) if hlen.sum() else np.zeros(0,np.int64); hT=np.concatenate(ht).astype(np.int64)
    hu=np.repeat(np.arange(n),hlen); pos_from_end=np.concatenate([np.arange(l)[::-1] for l in hlen]) if hlen.sum() else np.zeros(0)
    if 'f3' in fams:
        hc=np.bincount(hI,minlength=NI).astype(np.float32); col['f3_hpop']=hc[fl]/n*1e4
        wl=np.bincount(hI,weights=(pos_from_end<3)/(1.0+pos_from_end),minlength=NI).astype(np.float32); col['f3_hpop_l3']=wl[fl]/n*1e4
        l1=np.bincount(hI[pos_from_end==0],minlength=NI).astype(np.float32); col['f3_hpop_l1']=l1[fl]/n*1e4
        post=hT>=C; hp=np.bincount(hI[post],minlength=NI).astype(np.float32); col['f3_hpost']=hp[fl]/n*1e4
        hk=np.sort(hI*(1<<33)+hT); base=fl.astype(np.int64)*(1<<33); tr=np.repeat(ts,10).astype(np.int64)
        for nm,w in (('6h',6*3600),('24h',24*3600)):
            lo=np.searchsorted(hk,base+tr-w,'left'); hi_=np.searchsorted(hk,base+tr,'left'); col['f3_hrecent_'+nm]=(hi_-lo).astype(np.float32)   # others' clicks on j in the w hours before this event (visible in target histories)
            lo2=np.searchsorted(hk,base+tr,'left'); hi2=np.searchsorted(hk,base+tr+w,'right'); col['f3_hnext_'+nm]=(hi2-lo2).astype(np.float32)
        # own-history subtraction for the "post" count: own clicks of j in own history
        own=np.zeros(n*10,np.float32)
        for i in range(n):
            if hlen[i]:
                h=hi[i]; own[i*10:(i+1)*10]=(S[i][:,None]==h[None,:]).sum(1)
        col['f3_own_cnt']=own; col['f3_hpop_excl_own']=col['f3_hpop']-own/n*1e4
        for k in ('f3_hpop','f3_hpost','f3_hrecent_24h'):
            z,dm,rk=zrk(col[k]); col[k+'_z']=z; col[k+'_dm']=dm; col[k+'_rk']=rk
        col['f3_hpost_vs_train_ctr']=col['f3_hpost']/(ref['cl_all'][fl]/max(ref['nX'],1)*1e4+1.0)
    # ---- family 2: co-occurrence in target slates + last-click share among target users
    if 'f2' in fams:
        B=csr_matrix((np.ones(n*10,np.float32),(np.repeat(np.arange(n),10),fl)),shape=(n,NI)); CC=(B.T@B).tocsr()
        ri=np.repeat(S,10,axis=1).ravel(); ci=np.tile(S,(1,10)).ravel()   # all 100 pairs per event
        v=np.asarray(CC[ri,ci]).ravel().reshape(n,10,10)-1.0   # LOO own
        offd=~np.eye(10,dtype=bool); cj=(cnt[S]-1)   # (n,10)
        norm=np.sqrt((cj[:,:,None]+1)*(cj[:,None,:]+1))
        coo=(v/norm)[:,offd].reshape(n,10,9) if False else None
        vm=np.where(offd[None],v/norm,0).sum(2)/9; col['f2_cooc_norm']=vm.ravel()
        col['f2_cooc_raw']=(np.where(offd[None],v,0).sum(2)/9).ravel()
        # last-click share: users whose last click was h -> share of slates containing j
        last=np.array([x[-1] if len(x) else -1 for x in hi],np.int64); l3=[x[-3:] for x in hi]
        H=csr_matrix((np.ones((last>=0).sum(),np.float32),(np.flatnonzero(last>=0),last[last>=0])),shape=(n,NI)); A=(H.T@B).tocsr(); nh=np.asarray(H.sum(0)).ravel()
        lsh=np.full((n,10),np.nan,np.float32)
        ok=np.flatnonzero(last>=0)
        if len(ok):
            a=np.asarray(A[np.repeat(last[ok],10),S[ok].ravel()]).ravel().reshape(-1,10)-1.0; den=np.maximum(nh[last[ok]]-1,0)[:,None]
            lsh[ok]=np.where(den>0,a/np.maximum(den,1),np.nan)
            col['f2_lastshare_n']=np.repeat(np.where(nh[last]>0,nh[last]-1,np.nan),10).astype(np.float32)
        col['f2_lastshare']=lsh.ravel()
        # any of last-3 clicks (weighted share)
        H3=csr_matrix((np.concatenate([np.ones(len(x),np.float32) for x in l3]),(np.repeat(np.arange(n),[len(x) for x in l3]),np.concatenate(l3).astype(np.int64) if sum(len(x) for x in l3) else [])),shape=(n,NI)); A3=(H3.T@B).tocsr(); nh3=np.asarray(H3.sum(0)).ravel()
        s3=np.zeros((n,10),np.float32); d3=np.zeros(n,np.float32)
        for k in range(3):
            idx=[i for i in range(n) if len(l3[i])>k]
            if not idx: continue
            idx=np.array(idx); h=np.array([l3[i][-1-k] for i in idx],np.int64)
            a=np.asarray(A3[np.repeat(h,10),S[idx].ravel()]).ravel().reshape(-1,10)-1.0
            s3[idx]+=a; d3[idx]+=np.maximum(nh3[h]-1,0)
        col['f2_l3share']=np.where(d3[:,None]>0,s3/np.maximum(d3[:,None],1),np.nan).ravel()
        for k in ('f2_cooc_norm','f2_lastshare','f2_l3share'):
            z,dm,rk=zrk(col[k]); col[k+'_z']=z; col[k+'_dm']=dm; col[k+'_rk']=rk
    # ---- family 4: timing among target events
    if 'f4' in fams:
        o=np.argsort(ts,kind='stable'); rk_=np.empty(n); rk_[o]=np.arange(n); tss=ts[o]
        col['f4_trel_h']=rep((ts-C)/3600); col['f4_trank']=rep(rk_/n)
        col['f4_n_same_h']=rep(np.searchsorted(tss,ts+1800,'right')-np.searchsorted(tss,ts-1800,'left'))
        col['f4_n_6h']=rep(np.searchsorted(tss,ts+6*3600,'right')-np.searchsorted(tss,ts-6*3600,'left'))
        col['f4_n_before_frac']=rep(rk_/n)
        col['f4_hist_n']=rep(hlen)
    names=list(col); X=np.column_stack([np.asarray(col[k],np.float32) for k in names]).astype(np.float32)
    return X,names
def build_val(d,c,fams=FAMS):
    C=Jd(c); v=d.val(c); e=v['e'][::10]; S,ts,hi,ht=target_from_data(d,e)
    # sanity: slate order must match val row order
    S=v['item'].reshape(-1,10).astype(np.int64)   # slate in val-row order
    assert (np.sort(S,1)==np.sort(d.RI[e],1)).all()
    return build_tx(S,ts,hi,ht,C,train_ref(d,C),d.NI,fams)
if __name__=='__main__':
    d=D.Data()
    for c in [int(x) for x in sys.argv[1].split(',')]:
        X,names=build_val(d,c); np.save(f'/workspace/huawei/dw_lb/cache_tx_c{c}.npy',X); json.dump(names,open(f'/workspace/huawei/dw_lb/tx_names.json','w')); print(c,X.shape,flush=True)
