"""Public-score fit with graph as 11th component. cwd v21fit. python fitg.py [N]"""
import sys,os,pickle; sys.dont_write_bytecode=True
os.chdir('/workspace/huawei/v21fit'); sys.path.insert(0,'/workspace/huawei/v21fit')
import numpy as np
from addon import *      # sets, ix, ks, n, R0, dp, auc
from scipy.optimize import minimize
N=int(sys.argv[1]) if len(sys.argv)>1 else 80000
Gr=nr(np.load('/workspace/huawei/v21graph2/out/graph_eval_merged_rank.npy').astype(np.float64))
m=n+1; ig=n
idx=np.random.RandomState(0).choice(len(EN),200000,replace=False)
rg=np.array([np.corrcoef(Gr[idx],sets[k][idx])[0,1] for k in ks]); print('graph rho:',dict(zip(ks,rg.round(4))))
Rb=np.eye(m); Rb[:n,:n]=R0; Rb[ig,:n]=rg; Rb[:n,ig]=rg
pS1=np.load('pS1.npy'); pR=np.array([.136,.098,.5,-.03,-.015])
def mk(p):
    c,g,q,v8,v5=p; w=np.zeros(m); w[0]=1; w[1]=-c; w[ix['A']]=g*(1-q); w[ix['D']]=g*q; w[ix['v8r']]=v8; w[ix['v5']]=v5; return w
wQ2=mk([.136,.098,1,0,0]); wR1=mk(pR); wS1=mk(pS1)
# T1 weights as in recal3
p0=pickle.load(open('reopt_out.pkl','rb'))[1]; wT1=mk(pR+.667*(p0-pR))
rs=np.random.RandomState(77)
def prior10(Nb):
    ao=rs.uniform(.8185,.8205,Nb); a=np.zeros((Nb,n))
    a[:,ix['EN']]=.828856; a[:,ix['E0']]=.824558; a[:,ix['A']]=ao; a[:,ix['D']]=ao-rs.uniform(0,.0025,Nb)
    a[:,ix['P']]=rs.uniform(.800,.810,Nb); a[:,ix['v5']]=rs.uniform(.784,.794,Nb); a[:,ix['v8r']]=rs.uniform(.813,.8155,Nb)
    a[:,ix['S']]=ao+rs.uniform(-.001,.002,Nb); a[:,ix['slB']]=ao-rs.uniform(0,.0007,Nb); a[:,ix['listA']]=ao-rs.uniform(0,.0007,Nb)
    sd=np.where(R0>.99,.0008,.003); E=np.triu(rs.normal(size=(Nb,n,n)),1); E=E+E.transpose(0,2,1)
    R=np.minimum(R0[None]+E*sd[None],.9999); R[:,np.arange(n),np.arange(n)]=1
    return a,R
A10=[];R10=[];tot=0
while tot<60000:
    a_,R_=prior10(20000); k=np.linalg.eigvalsh(R_).min(1)>1e-6; A10.append(a_[k]); R10.append(R_[k]); tot+=k.sum()
A10=np.concatenate(A10); R10=np.concatenate(R10); Nb=len(A10); print('valid base draws',Nb)
def auc10(w,a,R): return auc((dp(a)@w[:n])/np.sqrt(np.einsum('i,nij,j->n',w[:n],R,w[:n])))-a[:,0]
P4=np.c_[auc10(wQ2,A10,R10),auc10(wR1,A10,R10),auc10(wS1,A10,R10),auc10(wT1,A10,R10)]; obs4=np.array([.000160,.000202,.000245,.000240]); obsG=.828955-.828856
tau=.00005; SIG=.00006
C4=SIG**2*np.eye(4)+tau**2*np.ones((4,4)); r4=obs4[None]-P4; ll=-.5*np.einsum('ni,ij,nj->n',r4,np.linalg.inv(C4),r4); W4=np.exp(ll-ll.max()); W4/=W4.sum(); print('stage1 ESS %.0f'%(1/np.sum(W4**2)))
base=np.random.RandomState(3).choice(Nb,6000,p=W4); K=20
a=np.repeat(A10[base],K,0); Rt=np.repeat(R10[base],K,0); N0=len(a)
a=np.c_[a,rs.uniform(.78,.84,N0)]; RR=np.zeros((N0,m,m)); RR[:,:n,:n]=Rt; 
rgn=np.minimum(rg[None]+rs.normal(size=(N0,n))*.004,.9999); RR[:,ig,:n]=rgn; RR[:,:n,ig]=rgn; RR[:,ig,ig]=1
ok=np.linalg.eigvalsh(RR).min(1)>1e-6; a=a[ok]; RR=RR[ok]; N=len(a); print('stage2 draws',N)
dd=dp(a); aEN=a[:,0]
def AUCw(w,d=dd,R=RR): return auc((d@w)/np.sqrt(np.einsum('i,nij,j->n',w,R,w)))
def GAIN(w): return AUCw(w)-aEN
def combw(wr,w0=wS1):
    W=np.repeat(w0[None],N,0)/(np.sqrt(np.einsum('i,nij,j->n',w0,RR,w0))[:,None]); W[:,ig]=wr; return W
def AUCW(W): return auc(np.einsum('ni,ni->n',W,dd)/np.sqrt(np.einsum('ni,nij,nj->n',W,RR,W)))
obs=np.r_[obs4,obsG]
P=np.c_[GAIN(wQ2),GAIN(wR1),GAIN(wS1),GAIN(wT1),AUCW(combw(-.045))-aEN]
res_all={}
for sig in (.00006,):
    tau=.00005; C=sig**2*np.eye(5)+tau**2*np.ones((5,5)); Ci=np.linalg.inv(C)
    res=obs[None]-P; C4_=C[:4,:4]; k_=C[4,:4]@np.linalg.inv(C4_); cm=res[:,:4]@k_; cv=C[4,4]-k_@C[:4,4]
    ll=-.5*(res[:,4]-cm)**2/cv; Wt=np.exp(ll-ll.max()); Wt/=Wt.sum(); ess=1/np.sum(Wt**2)
    bm=res.sum(1)*tau**2/(sig**2+5*tau**2); wm=lambda x:(Wt.reshape((-1,)+(1,)*(np.ndim(x)-1))*x).sum(0)
    def wq(x,q): o=np.argsort(x); c=np.cumsum(Wt[o]); return x[o][min(np.searchsorted(c,q),len(x)-1)]
    print(f'=== sig {sig} ESS {ess:.0f}')
    for nm,k in (('graph',ig),('aD',ix['D']),('ours',ix['A']),('aV5',ix['v5']),('aV8r',ix['v8r'])): print('%-6s post mean %.5f 5/95 %.5f %.5f'%(nm,wm(a[:,k]),wq(a[:,k],.05),wq(a[:,k],.95)))
    print('fit gains vs EN',dict(zip(('Q2','R1','S1','T1','G1'),(wm(P+bm[:,None])).round(6))),'obs',obs.round(6),'bias %.6f'%wm(bm))
    g0=AUCW(combw(0)); tab={}
    for w in (-.045,.0,.03,.05,.07,.09,.12,.15,.2,.25):
        g=AUCW(combw(w))-g0; tab[w]=g; print('  w=%+.3f gain vs S1 mean %+.6f 5/95 %+.6f %+.6f P>0 %.2f'%(w,wm(g),wq(g,.05),wq(g,.95),wm(g>0)))
    # finer grid optimum
    gr=np.arange(0,.31,.01); mg=np.array([wm(AUCW(combw(w))-g0) for w in gr]); wo=gr[mg.argmax()]; print('opt graph w (S1 fixed) %.2f gain %+.6f'%(wo,mg.max()))
    res_all[sig]=dict(wo=wo,tab={w:(wm(g),wq(g,.05),wq(g,.95),wm(g>0)) for w,g in tab.items()},Wt=Wt)
    # joint refit of (c,g,q,v8,v5) + graph weight on 6000 resample
    sel=np.random.RandomState(1).choice(N,6000,p=Wt); d6=dd[sel]; R6=RR[sel]
    def f6(x,full=True):
        p=np.r_[x[:5]] ; w0=mk(p); W=w0[None]/np.sqrt(np.einsum('i,nij,j->n',w0,R6,w0))[:,None]; W=np.repeat(W,1,0).copy(); W[:,ig]=x[5]
        return auc(np.einsum('ni,ni->n',W,d6)/np.sqrt(np.einsum('ni,nij,nj->n',W,R6,W)))
    base6=f6(np.r_[pS1,0.]); fo=lambda x:-(f6(x)-base6).mean()*1e4
    r=minimize(fo,np.r_[pS1,.1],method='Nelder-Mead',options={'maxiter':2500,'xatol':1e-4,'fatol':1e-9}); xo=r.x; xo[2]=min(max(xo[2],0),1)
    print('JOINT refit p(c,g,q,v8,v5)+wG=',np.round(xo,3),'gain vs S1 (resample) %+.6f'%(-fo(xo)*1e-4))
    fo2=lambda x:-(f6(np.r_[pS1,x[0]])-base6).mean()*1e4; r2=minimize(fo2,[.1],method='Nelder-Mead'); print('graph-only wG',np.round(r2.x,3),'gain %+.6f'%(-fo2(r2.x)*1e-4))
    res_all[sig]['joint']=xo
pickle.dump({k:{kk:vv for kk,vv in v.items() if kk!='Wt'} for k,v in res_all.items()},open('/workspace/huawei/v21graph2/out/fitg_res.pkl','wb'))
np.savez('/workspace/huawei/v21graph2/out/fitg_post.npz',a=a,Wt4=res_all[.00006]['Wt'],Wt6=res_all[.00006]['Wt'])
