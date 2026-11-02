import numpy as np, pickle
from scipy.stats import norm
from scipy.optimize import minimize, brentq
OUT='/workspace/huawei/cb_blend/ensig/'
Z=np.load(OUT+'eval_ranks_cache.npz'); K=[k for k in Z.files if k!='S1']
EV={k:Z[k].astype(np.float64) for k in K}; S1=Z['S1'].astype(np.float64)
dp=lambda a:np.sqrt(2)*norm.ppf(a); aucf=lambda d:norm.cdf(d/np.sqrt(2))
rs=np.random.RandomState(0).choice(len(S1),250000,replace=False)
names=['S1']+K; X=np.column_stack([S1[rs]]+[EV[k][rs] for k in K]); R=np.corrcoef(X.T); idx={n:i for i,n in enumerate(names)}
def ev_(a): return aucf(a)
def pred(wv,use,a):             # weights wv over members `use`, assumed AUCs a (dict) -> predicted combined AUC
    d=np.array([dp(a[u]) for u in use]); i=[idx[u] for u in use]; Rs=R[np.ix_(i,i)]; return aucf(wv@d/np.sqrt(wv@Rs@wv))
aS1=.829101
def best_add(a2,rho,lo=0,hi=.6):
    d1,d2=dp(aS1),dp(a2); f=lambda w:-aucf(((1-w)*d1+w*d2)/np.sqrt((1-w)**2+w**2+2*w*(1-w)*rho))
    ws=np.linspace(lo,hi,121); v=np.array([f(w) for w in ws]); j=v.argmin(); return ws[j],-v[j]-aS1
def need_auc(rho,gain):
    g=lambda a2:best_add(a2,rho)[1]-gain
    return brentq(g,.70,.9,xtol=1e-6) if g(.9)>0 else np.nan
# ---------- calibration of binormal model on known public scores ----------
pubA={'EN':.828856,'E0':.824558,'A_all':.8197,'D_dcn':.8189,'graph7s':.7951,'v8regen':.80,'v5':.80,'p33':.8052}
def cal(label,wd,actual,use_base='EN'):
    use=list(wd); w=np.array([wd[u] for u in use]); a=dict(pubA)
    return label,pred(w,use,a)-pubA['EN'],actual
print('--- calibration: predicted vs actual public gain over EN ---')
rows=[('Q2 EN-.136E0+.098D',{'EN':1,'E0':-.136,'D_dcn':.098},.829016-.828856),
 ('R1 EN-.136E0+.049A+.049D-.03v8-.015v5',{'EN':1,'E0':-.136,'A_all':.049,'D_dcn':.049,'v8regen':-.03,'v5':-.015},.829058-.828856),
 ('S1 recipe',{'EN':1,'E0':-.109,'A_all':.123*.335,'D_dcn':.123*.665,'v8regen':-.052,'v5':-.033},.829101-.828856)]
for l,wd,act in rows:
    use=list(wd); w=np.array([wd[u] for u in use]); a=dict(pubA); print(f'{l:45s} pred {pred(w,use,a)-pubA["EN"]:+.6f} actual {act:+.6f}')
# G2 / G1 on top of S1 via pairwise model with S1 as a single member:
rG=R[idx['S1'],idx['graph7s']]; print('rho(graph,S1)=',round(rG,4))
for w,act,l in [(.05,1.2e-5,'G2 S1+.05*nr(graph)'),(-.045,-1.46e-4,'G1 S1-.045*nr(graph)')]:
    d1,d2=dp(aS1),dp(pubA['graph7s']); ww=w/(1+w) if True else w   # nr(S1+w*G) ~ (1-u)S1+uG with u=w/(1+w) (both rank-scaled)
    ww=w/(1+w); p=aucf(((1-ww)*d1+ww*d2)/np.sqrt((1-ww)**2+ww**2+2*ww*(1-ww)*rG)); print(f'{l:25s} pred {p-aS1:+.6f} actual {act:+.6f}')
pickle.dump(dict(R=R,names=names),open(OUT+'R_eval.pkl','wb'))
