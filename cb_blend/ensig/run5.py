exec(open('run3.py').read().split('# ---------- calibration')[0])
import csv
st=pickle.load(open(OUT+'eval_stats.pkl','rb')); ROW=st['ROW']
LG=pickle.load(open(OUT+'local_gain.pkl','rb')); Rl=LG['R']
# public-AUC proxies: known public where available, else local C25 AUC (our family ~ public), else assumption
known={'EN':.828856,'E0':.824558,'p33':.8052,'ours_O(v21list B)':.8190,'A_all':.8197,'D_dcn':.8189}
alias={'ctl6(M)':'ctl6(M)'}
rows=[]
for k in K:
    if k=='EN': continue
    if k in known: a=known[k]; src='public'
    elif k in Rl and 25 in Rl[k]['aucs']: a=Rl[k]['aucs'][25]; src='localC25'
    elif k=='EBDISTILL': a=.8197; src='assumed(A)'
    else: a=.80; src='assumed'
    rS=ROW[k]['rho_S1']; rE=ROW[k]['rho_EN']
    d1=dp(aS1); margin=dp(a)/d1-rS
    w,g=best_add(a,rS)
    rows.append(dict(member=k,rho_EN=rE,rho_S1=rS,resid_sd=ROW[k]['resid_sd'],resid_ratio=ROW[k]['resid_ratio'],auc_proxy=a,auc_src=src,margin=margin,w_opt=w,binorm_gain_over_S1=g))
rows.sort(key=lambda r:-r['margin'])
for r in rows: print(f"{r['member']:20s} rEN {r['rho_EN']:.3f} rS1 {r['rho_S1']:.3f} resid_ratio {r['resid_ratio']:.3f} auc {r['auc_proxy']:.4f}({r['auc_src']}) margin {r['margin']:+.3f} wopt {r['w_opt']:.2f} gain {r['binorm_gain_over_S1']*1e4:+.2f}e-4")
pickle.dump(rows,open(OUT+'eval_table.pkl','wb'))
print('\n--- break-even: min AUC of X (rho to S1) for +gain over S1 (optimal weight, binormal) ---')
for g in (2e-4,6e-4):
    print('gain',g,{rho:round(need_auc(rho,g),4) for rho in (.80,.85,.90,.93,.95,.97,.98)})
print('\nS1 d-ratio vs rho: AUC needed for zero first-order gain = ',{rho:round(aucf(rho*dp(aS1)),4) for rho in (.85,.9,.93,.95,.97)})
