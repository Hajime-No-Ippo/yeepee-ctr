"""v18classic: classic CTR feature families (item freshness / audience match / user x time context / item x user-type CTR) on top of v17cooc A.
python mkv18.py NAME...   NAME in f1..f6,val23,val25,val26,eval.   Output feat_{fr,au,uc,ty}/NAME.parquet (rowid + float32 cols).
Leak policy (same as v12/v17cooc A): every aggregate uses TRAIN events with ts < X (X = fit cutoff of the table) + the click log rebuilt from them (D.snap(X));
per-row inputs = own user, own candidate/slate, own ts, own pre-ts history only.  No evaluation row label/impression/history is aggregated."""
import sys, os, time, warnings; sys.dont_write_bytecode=True; warnings.filterwarnings('ignore')
sys.path.insert(0,'/workspace/huawei/dw_p33'); sys.path.insert(0,'/workspace/huawei/v12')
from p33lib import *
from sigs import _zwithin, extend_eval
from mkdw import rows_train, CUT
O='/workspace/huawei/v18classic'; F32=np.float32
def zdm(x):
    z,dm=_zwithin(x); return z,dm
def rk(x):
    m=np.where(np.isnan(x.reshape(-1,10)),-np.inf,x.reshape(-1,10)); return m.argsort(1).argsort(1).astype(F32).ravel()
def flat(D,ev):
    st=D.OFF[ev]; ln=D.OFF[ev+1]-st; tot=int(ln.sum()); evi=np.repeat(np.arange(len(ev)),ln)
    idx=np.repeat(st-np.concatenate([[0],np.cumsum(ln)[:-1]]),ln)+np.arange(tot); return evi,idx,ln
def first_last(ids,vals,NI):
    """per-id min and max of vals (nan->inf/-inf if absent)"""
    f=np.full(NI,np.inf); l=np.full(NI,-np.inf)
    if len(ids)==0: return f,l
    o=np.lexsort((vals,ids)); i=ids[o]; v=vals[o].astype(np.float64)
    st=np.r_[True,i[1:]!=i[:-1]]; en=np.r_[i[1:]!=i[:-1],True]
    f[i[st]]=v[st]; l[i[en]]=v[en]; return f,l
HR=lambda t:(t%86400)/3600.0
DOW=lambda t:((t-J20)//86400)%7
def fam_fr(D,X,rows,S,nX):
    out={}; Sx=D.RI[:nX]; ts=D.ETS[:nX]; NI=D.NI; it=rows['item']; t=rows['ts'].astype(np.float64)
    fi,li=first_last(Sx.ravel(),np.repeat(ts,10),NI)                      # impressions (train events <X)
    fc,lc=first_last(S['ci'].astype(np.int64),S['ct'].astype(np.float64),NI)  # clicks (history appearances + clicked rows, <X)
    first=np.minimum(fi,fc); T0=first[np.isfinite(first)].min()
    age=np.where(np.isfinite(first[it]),(t-first[it])/3600,np.nan); out['fr_age_h']=age.astype(F32)
    out['fr_age_norm']=(age*3600/np.maximum(t-T0,1)).astype(F32)
    gi=np.where(np.isfinite(li[it]),(t-li[it])/3600,np.nan); gc=np.where(np.isfinite(lc[it]),(t-lc[it])/3600,np.nan)
    out['fr_gap_imp_h']=gi.astype(F32); out['fr_gap_clk_h']=gc.astype(F32)
    out['fr_gap_clk_rel']=(gc/np.maximum(age,1e-3)).astype(F32)
    for k in ('fr_age_h','fr_gap_imp_h','fr_gap_clk_h'):
        z,dm=zdm(out[k]); out[k+'_z']=z; out[k+'_dm']=dm; out[k+'_rk']=rk(out[k])
    return out
def user_stats(D,X,S,nX):
    NU=D.NU+1; eu=D.EU[:nX]; nev=np.bincount(eu,minlength=NU).astype(np.float64); clk=np.bincount(eu,weights=D.Y[:nX].sum(1),minlength=NU)
    p0=D.Y[:nX].sum()/nX; ctr=np.where(nev>0,(clk+5*p0)/(nev+5),np.nan)
    cu=S['cu'].astype(np.int64); n=np.bincount(cu,minlength=NU).astype(np.float64)
    f,l=first_last(cu,S['ct'].astype(np.float64),NU)
    rec=np.where(n>0,np.log1p((X-l)/3600),np.nan); gap=np.where(n>1,np.log1p((l-f)/np.maximum(n-1,1)/3600),np.nan)
    return dict(ctr=ctr,nclk=np.log1p(n),rec=rec,gap=gap)
def fam_au(D,X,rows,S,nX):
    out={}; NI=D.NI; NU=D.NU+1; us=user_stats(D,X,S,nX); ev=rows['e'][::10]; ue=D.EU[ev]; it=rows['item']
    key=np.unique(S['cu'].astype(np.int64)*NI+S['ci']); cu=key//NI; ci=(key%NI).astype(np.int64)     # unique (user,item) clickers
    n_i=np.bincount(ci,minlength=NI).astype(np.float64); m=5.0; Z=[]
    for s,v in us.items():
        x=v[cu]; ok=~np.isnan(x); xs=np.where(ok,x,0); cnt=np.bincount(ci,weights=ok,minlength=NI); s1=np.bincount(ci,weights=xs,minlength=NI); s2=np.bincount(ci,weights=xs*xs,minlength=NI)
        g=s1.sum()/cnt.sum(); gv=s2.sum()/cnt.sum()-g*g
        mu=(s1+m*g)/(cnt+m); var=(s2+m*(gv+g*g))/(cnt+m)-mu*mu; sd=np.sqrt(np.maximum(var,1e-6))
        u=np.repeat(v[ue],10); d=u-mu[it]; z=d/sd[it]; Z.append(z)
        out[f'au_{s}_d']=d.astype(F32); out[f'au_{s}_z']=z.astype(F32); out[f'au_{s}_dm']=zdm(d.astype(F32))[1]
    Zm=np.stack(Z,1); dist=np.sqrt(np.nanmean(Zm**2,1)); out['au_dist']=dist.astype(F32); out['au_dist_dm']=zdm(dist.astype(F32))[1]; out['au_dist_z']=zdm(dist.astype(F32))[0]
    out['au_nclk_item']=np.log1p(n_i[it]).astype(F32)
    return out
def circ(h,evi,n):
    a=2*np.pi*h/24; s=np.bincount(evi,weights=np.sin(a),minlength=n); c=np.bincount(evi,weights=np.cos(a),minlength=n); cnt=np.bincount(evi,minlength=n).astype(np.float64)
    R=np.sqrt(s*s+c*c)/np.maximum(cnt,1); mu=np.arctan2(s,c); return mu,np.where(cnt>0,R,np.nan),cnt
def cdist(h,mu):
    d=np.abs(2*np.pi*h/24-mu)%(2*np.pi); d=np.minimum(d,2*np.pi-d); return d*24/(2*np.pi)
def fam_uc(D,X,rows,S,nX):
    out={}; ev=rows['e'][::10]; n=len(ev); ts=rows['ts'][::10].astype(np.int64); NU=D.NU+1
    h_row=HR(ts); tb=(h_row//6).astype(int); dt=(DOW(ts)>=5).astype(int); jc=tb*2+dt; a=3.0
    def rep(x): return np.repeat(x,10).astype(F32)
    # --- own history clicks (<= 20)
    evi,idx,ln=flat(D,ev); ht=D.HT[idx]; hh=HR(ht); htb=(hh//6).astype(int); hdt=(DOW(ht)>=5).astype(int); hj=htb*2+hdt
    prior_c=np.bincount((HR(S['ct'])//6).astype(int)*2+(DOW(S['ct'])>=5).astype(int),minlength=8)/len(S['ct'])
    def share(own_j,own_evi,nown,rowj,prior,K):
        c=np.bincount(own_evi*K+own_j,minlength=n*K).reshape(n,K)[np.arange(n),rowj]; return (c+a*prior[rowj])/(nown+a)
    pj_tb=np.array([prior_c[2*b:2*b+2].sum() for b in range(4)]); pj_dt=np.array([prior_c[0::2].sum(),prior_c[1::2].sum()])
    sh_tb=share(htb,evi,ln,tb,pj_tb,4); sh_dt=share(hdt,evi,ln,dt,pj_dt,2); sh_j=share(hj,evi,ln,jc,prior_c,8)
    out['uc_tb_h']=rep(sh_tb); out['uc_dt_h']=rep(sh_dt); out['uc_tbdt_h']=rep(sh_j); out['uc_tb_h_lift']=rep(sh_tb/pj_tb[tb])
    mu,R,cnt=circ(hh,evi,n); out['uc_hdist_h']=rep(np.where(cnt>0,cdist(h_row,mu),np.nan)); out['uc_hR_h']=rep(R)
    # --- user's train events in snapshot (impressions' timestamps)
    eu=D.EU[:nX]; te=D.ETS[:nX]; eh=HR(te); etb=(eh//6).astype(int); edt=(DOW(te)>=5).astype(int); ej=etb*2+edt
    prior_e=np.bincount(ej,minlength=8)/nX; ue=D.EU[ev]
    cj=np.bincount(eu*8+ej,minlength=NU*8).reshape(NU,8).astype(np.float64); nu=cj.sum(1)
    ctb=cj.reshape(NU,4,2).sum(2); cdt=cj.reshape(NU,4,2).sum(1); pe_tb=np.array([prior_e[2*b:2*b+2].sum() for b in range(4)]); pe_dt=np.array([prior_e[0::2].sum(),prior_e[1::2].sum()])
    n_u=nu[ue]; s_tb=(ctb[ue,tb]+a*pe_tb[tb])/(n_u+a); s_dt=(cdt[ue,dt]+a*pe_dt[dt])/(n_u+a); s_j=(cj[ue,jc]+a*prior_e[jc])/(n_u+a)
    out['uc_tb_e']=rep(s_tb); out['uc_dt_e']=rep(s_dt); out['uc_tbdt_e']=rep(s_j)
    aa=2*np.pi*eh/24; ss=np.bincount(eu,weights=np.sin(aa),minlength=NU); cc=np.bincount(eu,weights=np.cos(aa),minlength=NU)
    mue=np.arctan2(ss,cc)[ue]; Re=np.sqrt(ss*ss+cc*cc)[ue]/np.maximum(n_u,1)
    out['uc_hdist_e']=rep(np.where(n_u>0,cdist(h_row,mue),np.nan)); out['uc_hR_e']=rep(np.where(n_u>0,Re,np.nan))
    return out
def fam_ty(D,X,rows,S,nX):
    out={}; NI=D.NI; Sx=D.RI[:nX]; Y=D.Y[:nX].astype(bool); it=rows['item']; ev=rows['e'][::10]; n=len(ev)
    def hrate(evs):
        ln=D.OFF[evs+1]-D.OFF[evs]; first=D.HT[np.minimum(D.OFF[evs],len(D.HT)-1)]; span=(D.ETS[evs]-first)/86400.0
        return np.where(ln>0,ln/(span+0.5),0.0)
    r_snap=hrate(np.arange(nX)); thr=np.median(r_snap); heavy_s=r_snap>=thr
    eu=D.EU[:nX]; o=np.argsort(eu,kind='stable'); es=eu[o]; st=np.r_[True,es[1:]!=es[:-1]]; grp=np.cumsum(st)-1; first_pos=np.where(st)[0]; ordn=np.empty(nX,np.int64); ordn[o]=np.arange(nX)-first_pos[grp]
    ret_s=ordn>0; nev=np.bincount(eu,minlength=D.NU+1)
    heavy_r=hrate(ev)>=thr; ret_r=nev[D.EU[ev]]>0
    p0=Y.sum()/Y.size; m=20.0
    imp=np.bincount(Sx.ravel(),minlength=NI).astype(np.float64); clk=np.bincount(Sx[Y],minlength=NI).astype(np.float64); ctr=(clk+m*p0)/(imp+m)
    def cls(mask):
        i=np.bincount(Sx[mask].ravel(),minlength=NI).astype(np.float64); c=np.bincount(Sx[mask[:,None]&Y],minlength=NI).astype(np.float64); return (c+m*ctr)/(i+m)
    ch,cl_,cn,cr=cls(heavy_s),cls(~heavy_s),cls(~ret_s),cls(ret_s)
    out['ty_ctr_heavy']=ch[it].astype(F32); out['ty_ctr_light']=cl_[it].astype(F32); out['ty_ctr_new']=cn[it].astype(F32); out['ty_ctr_ret']=cr[it].astype(F32)
    hr=np.repeat(heavy_r,10); rr=np.repeat(ret_r,10)
    vh=np.where(hr,ch[it],cl_[it]); vn=np.where(rr,cr[it],cn[it]); out['ty_ctr_h']=vh.astype(F32); out['ty_ctr_n']=vn.astype(F32)
    lh=np.log(vh/ctr[it]); ln_=np.log(vn/ctr[it]); out['ty_lift_h']=lh.astype(F32); out['ty_lift_n']=ln_.astype(F32)
    out['ty_lift_h_dm']=zdm(out['ty_lift_h'])[1]; out['ty_lift_n_dm']=zdm(out['ty_lift_n'])[1]; out['ty_ctr_h_dm']=zdm(out['ty_ctr_h'])[1]; out['ty_ctr_n_dm']=zdm(out['ty_ctr_n'])[1]
    out['ty_skew_hl']=np.log(ch[it]/cl_[it]).astype(F32); out['ty_skew_nr']=np.log(cn[it]/cr[it]).astype(F32)
    out['ty_utype']=np.repeat(heavy_r*2+ret_r,10).astype(F32)
    return out
FAM=dict(fr=fam_fr,au=fam_au,uc=fam_uc,ty=fam_ty)
def build(D,X,V,fams=('fr','au','uc','ty')):
    S=D.snap(X); nX=S['nX']; return {f:FAM[f](D,X,V,S,nX) for f in fams}
def rows_for(D,nm):
    X=CUT[nm]
    if nm=='eval': nt=D.NE; V=extend_eval(D); assert int(np.searchsorted(D.ETS,X))==nt
    else: V=rows_train(D,nm)
    return X,V
if __name__=='__main__':
    D=Data()
    for nm in sys.argv[1:]:
        t=time.time(); X,V=rows_for(D,nm); R=build(D,X,V)
        for f,o in R.items():
            df=pl.DataFrame({k:np.asarray(v,F32) for k,v in o.items()}).with_columns(pl.Series('rowid',V['rowid'])).select(['rowid']+list(o)); df.write_parquet(f'{O}/feat_{f}/{nm}.parquet')
        msg=f'{nm} X={X} rows={len(V["rowid"])} '+' '.join(f'{f}:{len(o)}' for f,o in R.items())+f' {round(time.time()-t)}s'; print(msg,flush=True); open(f'{O}/logs/mk.log','a').write(msg+'\n')
