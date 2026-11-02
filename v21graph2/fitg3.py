"""fitg3.py G8_merged_rank.npy dloc [tag]: public-score fit with G7 (old graph) + G8 (new graph) comps; obs Q2,R1,S1,T1,G1,G2. gain of G8 variants vs G2 and vs S1. cwd v21fit."""
import sys,os,pickle; sys.dont_write_bytecode=True
os.chdir('/workspace/huawei/v21fit'); sys.path.insert(0,'/workspace/huawei/v21fit')
import numpy as np
from addon import *
from scipy.stats import multivariate_normal
G8p=sys.argv[1]; dloc=float(sys.argv[2]); tag=sys.argv[3] if len(sys.argv)>3 else 'x'
G7=nr(np.load('/workspace/huawei/v21graph/out/graph_eval_merged_rank.npy').astype(np.float64)); G8=nr(np.load(G8p).astype(np.float64))
m=n+2; i7=n; i8=n+1
idx=np.random.RandomState(0).choice(len(EN),200000,replace=False)
r7=np.array([np.corrcoef(G7[idx],sets[k][idx])[0,1] for k in ks]); r8=np.array([np.corrcoef(G8[idx],sets[k][idx])[0,1] for k in ks]); r78=np.corrcoef(G7[idx],G8[idx])[0,1]
print('rho G8:',dict(zip(ks,r8.round(4))),'rho G7-G8 %.4f'%r78,'dloc',dloc)
pS1=np.load('pS1.npy'); pR=np.array([.136,.098,.5,-.03,-.015])
def mk(p):
    c,g,q,v8,v5=p; w=np.zeros(m); w[0]=1; w[1]=-c; w[ix['A']]=g*(1-q); w[ix['D']]=g*q; w[ix['v8r']]=v8; w[ix['v5']]=v5; return w
wQ2=mk([.136,.098,1,0,0]); wR1=mk(pR); wS1=mk(pS1); p0=pickle.load(open('reopt_out.pkl','rb'))[1]; wT1=mk(pR+.667*(p0-pR))
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
A10=np.concatenate(A10); R10=np.concatenate(R10); Nb=len(A10)
w10=lambda w:w[:n]
def auc10(w,a,R): return auc((dp(a)@w[:n])/np.sqrt(np.einsum('i,nij,j->n',w[:n],R,w[:n])))-a[:,0]
P4=np.c_[auc10(wQ2,A10,R10),auc10(wR1,A10,R10),auc10(wS1,A10,R10),auc10(wT1,A10,R10)]; obs4=np.array([.000160,.000202,.000245,.000240]); obsG=np.array([.828955-.828856,.829113-.828856])
tau=.00005; SIG=.00006
C4=SIG**2*np.eye(4)+tau**2*np.ones((4,4)); r4=obs4[None]-P4; ll=-.5*np.einsum('ni,ij,nj->n',r4,np.linalg.inv(C4),r4); W4=np.exp(ll-ll.max()); W4/=W4.sum()
base=np.random.RandomState(3).choice(Nb,6000,p=W4); K=20
a=np.repeat(A10[base],K,0); Rt=np.repeat(R10[base],K,0); N0=len(a)
a7=rs.uniform(.78,.84,N0); a8=a7+dloc+rs.normal(size=N0)*.002
a=np.c_[a,a7,a8]; RR=np.zeros((N0,m,m)); RR[:,:n,:n]=Rt
for ii,rr in ((i7,r7),(i8,r8)):
    rgn=np.minimum(rr[None]+rs.normal(size=(N0,n))*.004,.9999); RR[:,ii,:n]=rgn; RR[:,:n,ii]=rgn; RR[:,ii,ii]=1
x=np.minimum(r78+rs.normal(size=N0)*.002,.9999); RR[:,i7,i8]=x; RR[:,i8,i7]=x
ok=np.linalg.eigvalsh(RR).min(1)>1e-6; a=a[ok]; RR=RR[ok]; N=len(a); print('stage2 draws',N)
dd=dp(a); aEN=a[:,0]
def AUCW(W): return auc(np.einsum('ni,ni->n',W,dd)/np.sqrt(np.einsum('ni,nij,nj->n',W,RR,W)))
def comb(w7,w8,w0=wS1):
    W=np.repeat(w0[None],N,0)/(np.sqrt(np.einsum('i,nij,j->n',w0,RR,w0))[:,None]); W[:,i7]=w7; W[:,i8]=w8; return W
def GAIN(w): return AUCW(np.repeat(w[None],N,0)/np.sqrt(np.einsum('i,nij,j->n',w,RR,w))[:,None])-aEN
P=np.c_[GAIN(wQ2),GAIN(wR1),GAIN(wS1),GAIN(wT1),AUCW(comb(-.045,0))-aEN,AUCW(comb(.05,0))-aEN]; obs=np.r_[obs4,obsG]
C=SIG**2*np.eye(6)+tau**2*np.ones((6,6)); res=obs[None]-P
C11=C[:4,:4]; C21=C[4:,:4]; k_=C21@np.linalg.inv(C11); cm=res[:,:4]@k_.T; cv=C[4:,4:]-k_@C21.T; ci=np.linalg.inv(cv)
d=res[:,4:]-cm; ll=-.5*np.einsum('ni,ij,nj->n',d,ci,d); Wt=np.exp(ll-ll.max()); Wt/=Wt.sum(); print('ESS %.0f'%(1/np.sum(Wt**2)))
wm=lambda x:(Wt.reshape((-1,)+(1,)*(np.ndim(x)-1))*x).sum(0)
def wq(x,q): o=np.argsort(x); c=np.cumsum(Wt[o]); return x[o][min(np.searchsorted(c,q),len(x)-1)]
for nm,k in (('G7',i7),('G8',i8),('aD',ix['D']),('ours',ix['A'])): print('%-5s post mean %.5f 5/95 %.5f %.5f'%(nm,wm(a[:,k]),wq(a[:,k],.05),wq(a[:,k],.95)))
bm=res.sum(1)*tau**2/(SIG**2+6*tau**2)
print('fit gains vs EN',dict(zip(('Q2','R1','S1','T1','G1','G2'),wm(P+bm[:,None]).round(6))),'obs',obs.round(6))
gS1=AUCW(comb(0,0)); gG2=AUCW(comb(.05,0))
print('G2 vs S1 pred %+.6f'%wm(gG2-gS1))
def show(lab,w7,w8):
    g=AUCW(comb(w7,w8))
    gv=g-gG2; gs=g-gS1
    print('%-18s vs G2 mean %+.6f 5/95 %+.6f %+.6f P>0 %.2f P>5e-5 %.2f | vs S1 mean %+.6f'%(lab,wm(gv),wq(gv,.05),wq(gv,.95),wm(gv>0),wm(gv>5e-5),wm(gs)),flush=True); return wm(gv)
print('--- G8 replaces G7 (w7=0):')
for w in (.03,.05,.07,.1,.15,.2): show('w8=%.2f'%w,0,w)
print('--- G8 on top of G2 (w7=.05):')
for w in (.02,.05,.1): show('w7=.05 w8=%.2f'%w,.05,w)
best=max(((show('grid',w7,w8) if False else wm(AUCW(comb(w7,w8))-gG2),w7,w8) for w7 in (0,.025,.05) for w8 in (0,.02,.04,.06,.08,.1,.12,.15)))
print('best grid (gain vs G2, w7,w8):',best); show('best',best[1],best[2])
