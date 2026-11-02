s=open('/workspace/huawei/cb_blend/ensig/run2.py').read(); exec(s.split('R={}\n')[0])
import pickle
LG=pickle.load(open(OUT+'local_gain.pkl','rb')); R=LG['R']; WG=LG['WG']
# ---- held-out single-member gain (w chosen on the other cut) ----
HO={}
for k,r in R.items():
    cv=r['curve']
    if 23 in cv and 25 in cv:
        w25=int(np.argmax(cv[25])); w23=int(np.argmax(cv[23]))
        HO[k]=dict(g23_w25=cv[23][w25]*1e4,g25_w23=cv[25][w23]*1e4,w23=WG[w23],w25=WG[w25])
    elif 25 in cv and 26 in cv:
        w25=int(np.argmax(cv[25])); HO[k]=dict(g26_w25=cv[26][w25]*1e4,g25_best=cv[25].max()*1e4,w25=WG[w25])
for k,v in HO.items(): print(k,{a:round(float(b),2) for a,b in v.items()})
# ---- greedy forward, fit on one cut, test on the other (nonneg weights; each step adds member at best w, shrink base) ----
cand=[k for k in R if 23 in R[k]['aucs'] and 25 in R[k]['aucs'] and k not in('ctl6(M)','big(v21seq)','graph7s','list(v21list)','z_lgb(nmAs)')]
X={c:{k:rk(getloc(k,c)) for k in cand} for c in (23,25)}
def greedy(fit,test,steps=4):
    cur={c:base[c].copy() for c in (23,25)}; log=[]
    for s in range(steps):
        best=None
        for k in cand:
            for w in (.03,.06,.1,.15,.2):
                a=auc(fit,(1-w)*cur[fit]+w*X[fit][k])
                if best is None or a>best[0]: best=(a,k,w)
        a,k,w=best
        for c in (23,25): cur[c]=(1-w)*cur[c]+w*X[c][k]
        log.append((k,w,(a-B0[fit])*1e4,(auc(test,cur[test])-B0[test])*1e4)); print(fit,'->',test,log[-1],flush=True)
    return log
G1=greedy(23,25); G2=greedy(25,23)
pickle.dump(dict(HO=HO,G1=G1,G2=G2),open(OUT+'heldout.pkl','wb'))
