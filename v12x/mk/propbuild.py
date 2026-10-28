import numpy as np
from common import *
def build(d,C,v):
    e=v['e'][::10]; nE=len(e)
    ec=d.Y.max(1).astype(np.float32); ts=d.ETS; u=d.EU
    nX=int(np.searchsorted(ts,C))  # events before C
    # per-user aggregates over events with ts<C
    n_ev=np.bincount(u[:nX],minlength=d.NU).astype(np.float32); n_cl=np.bincount(u[:nX],weights=ec[:nX],minlength=d.NU)
    first=np.full(d.NU,np.inf); np.minimum.at(first,u[:nX],ts[:nX]); last=np.full(d.NU,-np.inf); np.maximum.at(last,u[:nX],ts[:nX])
    # last-3 events click rate before C, trailing no-click streak
    o=np.lexsort((ts[:nX],u[:nX])); us=u[:nX][o]; ecs=ec[:nX][o]; tss=ts[:nX][o]
    b=np.r_[0,np.flatnonzero(np.diff(us))+1,len(us)]
    l3=np.full(d.NU,np.nan,np.float32); streak=np.zeros(d.NU,np.float32); gapm=np.full(d.NU,np.nan,np.float32); gapsd=np.full(d.NU,np.nan,np.float32)
    for a,bb in zip(b[:-1],b[1:]):
        uu=us[a]; seg=ecs[a:bb]; l3[uu]=seg[-3:].mean(); r=0
        for x in seg[::-1]:
            if x==0: r+=1
            else: break
        streak[uu]=r
        if bb-a>=2: g=np.diff(tss[a:bb]); gapm[uu]=g.mean(); gapsd[uu]=g.std()
    ue=u[e]; tev=ts[e]
    shrink=(n_cl[ue]+0.63*3)/(n_ev[ue]+3)
    days=(C-J20)/DAY
    col={}
    col['p_nev']=n_ev[ue]; col['p_evrate']=n_ev[ue]/days; col['p_ctr']=np.where(n_ev[ue]>0,n_cl[ue]/np.maximum(n_ev[ue],1),np.nan); col['p_ctr_s']=shrink
    col['p_l3']=l3[ue]; col['p_streak']=streak[ue]; col['p_gapm']=gapm[ue]/3600; col['p_gapsd']=gapsd[ue]/3600
    col['p_since_ev']=np.where(np.isfinite(last[ue]),(tev-last[ue])/3600,np.nan); col['p_since_first']=np.where(np.isfinite(first[ue]),(tev-first[ue])/3600,np.nan)
    col['p_gap_ratio']=col['p_since_ev']/(col['p_gapm']+1)
    # own history
    hl=d.Hlen[e]; col['h_len']=hl.astype(np.float32)
    hmin=np.full(nE,np.nan); hmax=np.full(nE,np.nan); hgm=np.full(nE,np.nan); hgsd=np.full(nE,np.nan); hglast=np.full(nE,np.nan); h1=np.zeros(nE); h3=np.zeros(nE); h7=np.zeros(nE); hwarm=np.zeros(nE); hwin=np.zeros(nE); huniq=np.zeros(nE)
    for i,ee in enumerate(e):
        a,bb=d.OFF[ee],d.OFF[ee+1]
        if bb-a==0: continue
        t=d.HT[a:bb]; it=d.HI[a:bb]; tt=tev[i]
        hmin[i]=(tt-t[0])/3600; hmax[i]=(tt-t[-1])/3600; huniq[i]=len(set(it.tolist()))
        h1[i]=(t>tt-DAY).sum(); h3[i]=(t>tt-3*DAY).sum(); h7[i]=(t>tt-7*DAY).sum(); hwarm[i]=(t<J20).sum(); hwin[i]=(t>=J20).sum()
        if bb-a>=2: g=np.diff(t); hgm[i]=g.mean()/3600; hgsd[i]=g.std()/3600; hglast[i]=g[-1]/3600
    col.update(h_span=hmin,h_since_last=hmax,h_gap_mean=hgm,h_gap_sd=hgsd,h_gap_last=hglast,h_1d=h1,h_3d=h3,h_7d=h7,h_warm=hwarm,h_win=hwin,h_uniq=huniq,h_uniq_ratio=huniq/np.maximum(hl,1))
    col['h_rate_day']=np.where(hl>=2,(hl-1)/np.maximum(hmin-hmax,1e-3)*24,np.nan)
    # clicks per event: window clicks (in train, ts<C) / events
    col['p_clk_per_ev']=n_cl[ue]/np.maximum(n_ev[ue],1)
    # history density per unit time vs event rate: implied clicks per event = click_rate/event_rate
    erate=np.where(n_ev[ue]>0,n_ev[ue]/np.maximum((C-J20)/3600/24,1e-3),np.nan)  # events per day
    col['implied_cpe']=col['h_rate_day']/(erate+0.5)
    col['gap_over_hgap']=col['p_gapm']/(col['h_gap_mean']+1)
    X=np.column_stack([np.repeat(np.asarray(col[k],np.float32),10) for k in col]).astype(np.float32)
    return X,list(col)

