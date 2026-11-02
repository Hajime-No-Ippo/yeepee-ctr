import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[k]='1'
import numpy as np
from scipy.stats import norm
Zs=np.load('../u12/arrs.npz'); V='/workspace/huawei/v21fit/'
G=np.load('/workspace/huawei/v21graph/out/graph_eval_merged_rank.npy').astype(np.float64)
G=(np.argsort(np.argsort(G))/(len(G)-1.))
C=[Zs['EN'],Zs['E0'],Zs['A'],Zs['D'],np.load(V+'inv_v8regen.npy'),np.load(V+'inv_v5.npy'),G]
X=np.column_stack(C).astype(np.float64); Xc=X-X.mean(0); Cov=Xc.T@Xc/len(X)
pi=.0631; kap=1.1328
def wv(c,g,q,v8,v5): return np.array([1,-c,g*(1-q),g*q,v8,v5,0.])
wS1=wv(.10902810947726392,.12251977564719992,.665,-.052490334215038625,-.033138516122339604)
sraw=np.sqrt(wS1@Cov@wS1); s=sraw/np.sqrt(1/12); print('sigma_raw S1',sraw,'scale',s)
def nrw(wraw,g=0.):
    w=wraw.copy(); w[6]+=g*s; return w
W={'EN':np.array([1,0,0,0,0,0,0.]),'Q2':np.array([1,-.136,0,.098,0,0,0]),'R1':np.array([1,-.136,.049,.049,-.03,-.015,0]),
   'S1':wS1,'T1':wv(.081,.148,.834,-.075,-.052),'G1':nrw(wS1,-.045)}
pub={'EN':.828856,'Q2':.829016,'R1':.829058,'S1':.829101,'T1':.829096,'G1':.828955}
rng=np.random.default_rng(3); N=1_500_000
pm=np.array([.8194,.8181,.8146,.7925]); psd=np.array([.004,.004,.006,.008])
S=pm+rng.normal(size=(N,4))*psd; aG=rng.uniform(.775,.825,N); b=rng.normal(size=N)*.00005
a=np.column_stack([np.full(N,.828856),np.full(N,.824558),S,aG])-.5
def auc(wt):
    delta=a@wt; var=wt@Cov@wt; sw2=var-pi*(1-pi)*delta**2; return norm.cdf(kap*delta/np.sqrt(sw2)/np.sqrt(2))
M={k:auc(w) for k,w in W.items()}
def wq(x,wt,q): o=np.argsort(x); c=np.cumsum(wt[o]); return np.interp(q,c,x[o])
wf=np.linspace(0,.5,26); gg=np.array([auc(nrw(wS1,w))-M['S1'] for w in wf])
ws=[0,.03,.05,.075,.1,.15,.2,.3]; gi=[int(round(w/.02)) for w in ws]
for sig in (.00004,.00005,.00006):
    ll=sum(-.5*((pub[k]-b-M[k])/sig)**2 for k in pub); wt=np.exp(ll-ll.max()); wt/=wt.sum(); ess=1/np.sum(wt**2)
    print(f'\nsig {sig}: ESS {ess:.0f}')
    print(' post mean aA,aD,aV8,aV5,aG',np.round(wt@np.column_stack([S,aG]),4))
    print(f' graph eval AUC mean {wt@aG:.4f}  90% [{wq(aG,wt,.05):.4f},{wq(aG,wt,.95):.4f}]')
    for w in ws:
        g=np.array([auc(nrw(wS1,w))-M['S1']])[0]
        print(f'  w={w:.3f}: gain vs S1 mean {wt@g*1e5:+.1f}e-5 [{wq(g,wt,.05)*1e5:+.1f},{wq(g,wt,.95)*1e5:+.1f}] P(>0) {wt@(g>0):.2f} P(>=+5e-5) {wt@(g>=5e-5):.2f} -> public {.829101+wt@g:.6f}')
    ws_=wf[gg.argmax(0)]
    print(' w* per-draw mean',round(wt@ws_,3),'90%',[round(wq(ws_,wt,.05),3),round(wq(ws_,wt,.95),3)],'P(w*>=.05)',round(wt@(ws_>=.05),2),' w maximizing mean gain',wf[(gg@wt).argmax()],'mean gain there',round((gg@wt).max()*1e5,1))
    print(' G1 pred public gain vs S1 (post mean)',round(wt@(M['G1']-M['S1'])*1e5,1),'e-5 (obs -14.6e-5)')
