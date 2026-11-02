exec(open('/workspace/huawei/cb_blend/ensig/ensig.py').read())
import time, pickle
t0=time.time()
# ---------- eval vectors + alignment check ----------
EV={}; INV=[]
for k,r in REG.items():
    v=r['ev'](); ns=None
    if isinstance(v,tuple): v,ns=v
    v=np.asarray(v,np.float64); assert v.shape==(N,) and np.isfinite(v).all(),k
    EV[k]=nr(v)
    # alignment check: rho vs EN in merged order, and vs a deliberately shuffled-order control (sanity: must be ~0)
    rho=np.corrcoef(EV[k],EN)[0,1]; rsh=np.corrcoef(EV[k][np.random.RandomState(1).permutation(N)],EN)[0,1]
    INV.append((k,r['fam'],ns,rho,rsh)); print(f'{k:20s} nseeds={ns} rho_EN={rho:.4f} (shuffled ctrl {rsh:+.4f})',flush=True)
EV['EN']=EN
# ---------- stats vs EN and S1 ----------
sdU=np.sqrt(1/12.)
X2=np.column_stack([np.ones(N),EN,S1]); Q,_=np.linalg.qr(X2)
ROW={}
for k,v in EV.items():
    rEN=np.corrcoef(v,EN)[0,1]; rS1=np.corrcoef(v,S1)[0,1]
    res=v-Q@(Q.T@v); rs=res.std(); ratio=rs/v.std()
    # partial corr of member with S1 given EN? residual of member on EN only
    ROW[k]=dict(rho_EN=rEN,rho_S1=rS1,resid_sd=rs,resid_ratio=ratio,R2=1-ratio**2)
print('stats done',time.time()-t0)
pickle.dump(dict(INV=INV,ROW=ROW),open(OUT+'eval_stats.pkl','wb'))
np.savez(OUT+'eval_ranks_cache.npz',**{k.replace('/','_'):v.astype(np.float32) for k,v in EV.items()},S1=S1.astype(np.float32))
