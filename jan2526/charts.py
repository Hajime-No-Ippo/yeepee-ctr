import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
pd_=pd.read_csv('per_day.csv'); it=pd.read_csv('items_per_day.csv'); v=pd.read_csv('valsets.csv',index_col=0)
hm=pd.read_csv('hour_mix.csv',index_col=0); D=pd.read_pickle('val_dists.pkl'); sa=pd.read_csv('signal_auc.csv')
lab=pd_.day.tolist(); col=['#4a7bd0']*5+['#e07b39']*2+['#555555']
# 1 per-day panel
f,ax=plt.subplots(2,3,figsize=(15,8))
ax[0,0].bar(lab,pd_.events,color=col); ax[0,0].set_title('Events per day (eval = 1 per user)')
ax[0,1].bar(lab[:7],pd_.ev_click_rate[:7],color=col[:7]); ax[0,1].set_ylim(.6,.65); ax[0,1].set_title('Event click rate (≥1 click in 10)')
x=np.arange(7); ax[0,2].bar(x-.2,pd_.ctr_inhist[:7],.4,label='candidate in history',color='#2a9d8f'); ax[0,2].bar(x+.2,pd_.ctr_notinhist[:7]*5,.4,label='not in history (×5)',color='#b0b0b0')
ax[0,2].set_xticks(x,lab[:7]); ax[0,2].legend(fontsize=8); ax[0,2].set_title('Row CTR by in-history flag')
ax[1,0].plot(lab,pd_.hlen_mean,'o-'); ax[1,0].set_title('Mean history length'); a2=ax[1,0].twinx(); a2.plot(lab,pd_.warm_share,'s--',color='#e07b39'); a2.set_ylabel('warm-up share of history',color='#e07b39')
ax[1,1].bar(lab,pd_.first_day_in_window_share,color=col); ax[1,1].set_title('Share of the day\'s users first seen in train that day'); ax[1,1].set_yscale('log')
ax[1,2].bar(lab[:7],it.js_imp_vs_wd[:7],color=col[:7],label='day vs other weekdays'); ax[1,2].axhline(it.js_imp_vs_wd[7],color='k',ls='--',label='eval vs Mon–Fri'); ax[1,2].axhline(0.0203,color='r',ls=':',label='noise floor at eval size')
ax[1,2].set_title('Item impression JS divergence'); ax[1,2].legend(fontsize=8)
f.suptitle('Jan 20–26 by day: Sat 25 / Sun 26 (orange) look like weekdays; trends are window position, not weekend',fontsize=12); f.tight_layout(); f.savefig('01_per_day.png',dpi=110)
# 2 hour mix
f,ax=plt.subplots(1,2,figsize=(14,4.5))
ax[0].plot(hm.index,hm[['0','1','2','3','4']].mean(1)*100,label='Mon–Fri (all events)'); ax[0].plot(hm.index,hm['5']*100,label='Sat 25'); ax[0].plot(hm.index,hm['6']*100,label='Sun 26'); ax[0].plot(hm.index,hm['7']*100,'k--',label='EVAL (first event/user)')
ax[0].set_xlabel('UTC hour'); ax[0].set_ylabel('% of events'); ax[0].legend(); ax[0].set_title('Hour-of-day mix: all train events vs eval')
for C,s in [(24,'-'),(25,'-'),(26,'-'),(27,'k--')]:
    h=np.bincount(D[C].hr,minlength=24)/len(D[C])*100; ax[1].plot(range(24),h,s,label='EVAL' if C==27 else f'val C={C}')
ax[1].set_xlabel('UTC hour'); ax[1].legend(); ax[1].set_title('Hour mix of first-event-per-user sets')
f.tight_layout(); f.savefig('02_hour_mix.png',dpi=110)
# 3 val vs eval
f,ax=plt.subplots(1,2,figsize=(14,4.8))
for C,s in [(23,'-'),(24,'-'),(25,'-'),(26,'-'),(27,'k--')]:
    st=np.sort(D[C].stale_h); ax[0].plot(st,np.arange(1,len(st)+1)/len(st),s,label='EVAL' if C==27 else f'val C={C}')
ax[0].set_xlabel('hours since cutoff'); ax[0].set_ylabel('CDF'); ax[0].set_xlim(0,170); ax[0].legend(); ax[0].set_title('Staleness: C=25/26 are truncated at 48h/24h')
m=['weekend_share','stale_gt24h','new_user_share','warm_share','hlen20_share','empty_hist']
sets=['val C=23','val C=24','val C=25','val C=26','EVAL']; w=.16
for j,sname in enumerate(sets):
    ax[1].bar(np.arange(len(m))+(j-2)*w, v.loc[sname,m], w, label=sname, color=(None if sname!='EVAL' else 'k'))
ax[1].set_xticks(np.arange(len(m)),['weekend','stale≥24h','new user','warm-up share','hist=20','empty hist'],fontsize=9); ax[1].legend(fontsize=8); ax[1].set_title('Validation sets vs eval')
f.tight_layout(); f.savefig('03_val_vs_eval.png',dpi=110)
# 4 signal auc
f,ax=plt.subplots(1,2,figsize=(14,4.5))
dd=sa[sa.set.str.startswith('day')]; c4=['#4a7bd0']*4+['#e07b39']*2
ax[0].bar(dd.set,dd.auc_sim,color=c4); ax[0].set_ylim(.72,.78); ax[0].set_title('History-similarity proxy (SVD-16 cosine) AUC by day\nfit on clicks before each day; weekend = orange')
for i,r in enumerate(dd.itertuples()): ax[0].text(i,r.auc_sim+.001,f'{r.auc_sim:.3f}\ninh {r.auc_inhist:.3f}\npop {r.auc_pop:.3f}',ha='center',fontsize=7)
vv=sa[sa.set.isin(['val C=24 stale<24h','val C=24 stale>=24h','val C=25 stale<24h','val C=25 stale>=24h','val C=26'])]
ax[1].bar([s.replace('val ','') for s in vv.set],vv.auc_sim,color=['#4a7bd0','#e07b39','#e07b39','#e07b39','#e07b39']); ax[1].set_ylim(.72,.78)
for i,r in enumerate(vv.itertuples()): ax[1].text(i,r.auc_sim+.001,f'{r.auc_sim:.3f}',ha='center',fontsize=8)
ax[1].set_title('Same proxy on validation sets, split by staleness\n(drop comes from staleness, also inside all-weekend C=25)'); ax[1].tick_params(axis='x',labelsize=8)
f.tight_layout(); f.savefig('04_signal_auc.png',dpi=110)
