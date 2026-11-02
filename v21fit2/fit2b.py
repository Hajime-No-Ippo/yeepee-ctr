"""fit2.py: public-score binormal fit, 12 comps (10 base + G7 + G8), obs Q2,R1,S1,T1,G1,G2,G3 (gains vs EN).
usage: python fit2.py [sig_graph] ; writes post2.npz (resampled posterior draws)"""
import sys,os; sys.dont_write_bytecode=True
sys.path.insert(0,'/workspace/huawei/v21fit')
import numpy as np
from lib import *
from binormal import dp,auc
H='/workspace/huawei/'; os.chdir(H+'v21fit2')
SIG=float(os.environ.get('SIG','.00006')); tau=float(os.environ.get('TAU','.00005'))
A=np.load(H+'v21fit/A_all.npy'); D=np.load(H+'v21fit/D_dcn.npy')
sets={'EN':EN,'E0':E0,'A':A,'D':D,'P':P,'v5':np.load(H+'v21fit/inv_v5.npy'),'v8r':np.load(H+'v21fit/inv_v8regen.npy'),'S':np.load(H+'v21fit/S_ebdistill.npy'),'slB':np.load(H+'v21fit/inv_v21slateB.npy'),'listA':np.load(H+'v21fit/inv_v21listA.npy')}
ks=list(sets); n=len(ks); ix={k:i for i,k in enumerate(ks)}
G7=nr(np.load(H+'v21graph/out/graph_eval_merged_rank.npy').astype(np.float64)); G8=nr(np.load(H+'v21graph2/out/graph2s3b_f6_eval_merged_rank.npy').astype(np.float64))
X=np.c_[[nr(sets[k]) for k in ks]+[G7,G8]]
idx=np.random.RandomState(0).choice(len(EN),200000,replace=False)
Rfull=np.corrcoef(X[:,idx]); np.save('R12.npy',Rfull)
del X
R0=Rfull[:n,:n]; r7=Rfull[n,:n]; r8=Rfull[n+1,:n]; r78=Rfull[n,n+1]
print('rho G7',dict(zip(ks,r7.round(4))),'\nrho G8',dict(zip(ks,r8.round(4))),'r78 %.4f'%r78,flush=True)
m=n+2; i7=n; i8=n+1
pS1=np.load(H+'v21fit/pS1.npy'); pR=np.array([.136,.098,.5,-.03,-.015])
def mk(p,m=m):
    c,g,q,v8,v5=p; w=np.zeros(m); w[0]=1; w[1]=-c; w[ix['A']]=g*(1-q); w[ix['D']]=g*q; w[ix['v8r']]=v8; w[ix['v5']]=v5; return w
p0=__import__('pickle').load(open(H+'v21fit/reopt_out.pkl','rb'))[1]
wQ2=mk([.136,.098,1,0,0]); wR1=mk(pR); wS1=mk(pS1); wT1=mk(pR+.667*(p0-pR))
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
def auc10(w,a,R): return auc((dp(a)@w[:n])/np.sqrt(np.einsum('i,nij,j->n',w[:n],R,w[:n])))-a[:,0]
P4=np.c_[[auc10(w,A10,R10) for w in (wQ2,wR1,wS1,wT1)]].T
obs4=np.array([.000160,.000202,.000245,.000240]); obsG=np.array([.828955,.829113,.829117])-.828856
C4=SIG**2*np.eye(4)+tau**2*np.ones((4,4)); r4=obs4[None]-P4; ll=-.5*np.einsum('ni,ij,nj->n',r4,np.linalg.inv(C4),r4); W4=np.exp(ll-ll.max()); W4/=W4.sum()
print('stage1 ESS %.0f'%(1/np.sum(W4**2)),flush=True)
base=np.random.RandomState(3).choice(Nb,8000,p=W4); K=25
a=np.repeat(A10[base],K,0); Rt=np.repeat(R10[base],K,0); N0=len(a)
a7=rs.uniform(.795,.82,N0); dloc=rs.uniform(-.003,.009,N0); a8=a7+dloc
a=np.c_[a,a7,a8]; RR=np.zeros((N0,m,m)); RR[:,:n,:n]=Rt
for ii,rr in ((i7,r7),(i8,r8)):
    rgn=np.minimum(rr[None]+rs.normal(size=(N0,n))*.004,.9999); RR[:,ii,:n]=rgn; RR[:,:n,ii]=rgn; RR[:,ii,ii]=1
x=np.minimum(r78+rs.normal(size=N0)*.002,.9999); RR[:,i7,i8]=x; RR[:,i8,i7]=x
ok=np.linalg.eigvalsh(RR).min(1)>1e-6; a=a[ok]; RR=RR[ok]; N=len(a); print('stage2 draws',N,flush=True)
dd=dp(a); aEN=a[:,0]
def AUCW(W): return auc(np.einsum('ni,ni->n',W,dd)/np.sqrt(np.einsum('ni,nij,nj->n',W,RR,W)))
sS1=np.sqrt(np.einsum('i,nij,j->n',wS1,RR,wS1))
def comb(w7,w8,w0=None):
    w0=wS1 if w0 is None else w0; W=np.repeat(w0[None],N,0)/sS1[:,None]; W[:,i7]=w7; W[:,i8]=w8; return W
def GAINW(W): return AUCW(W)-aEN
def GAIN(w): return AUCW(np.repeat(w[None],N,0))-aEN
P=np.c_[GAIN(wQ2),GAIN(wR1),GAIN(wS1),GAIN(wT1),GAINW(comb(-.045,0)),GAINW(comb(.05,0)),GAINW(comb(0,.05))]
obs=np.r_[obs4,obsG]; k_=len(obs)
C=SIG**2*np.eye(k_)+tau**2*np.ones((k_,k_)); res=obs[None]-P
C11=C[:4,:4]; C21=C[4:,:4]; kk=C21@np.linalg.inv(C11); cm=res[:,:4]@kk.T; cv=C[4:,4:]-kk@C21.T; ci=np.linalg.inv(cv)
d=res[:,4:]-cm; ll=-.5*np.einsum('ni,ij,nj->n',d,ci,d); Wt=np.exp(ll-ll.max()); Wt/=Wt.sum(); print('ESS %.0f'%(1/np.sum(Wt**2)),flush=True)
wm=lambda x:(Wt.reshape((-1,)+(1,)*(np.ndim(x)-1))*x).sum(0)
def wq(x,q): o=np.argsort(x); c=np.cumsum(Wt[o]); return x[o][min(np.searchsorted(c,q),len(x)-1)]
for nm,kx in (('G7',i7),('G8',i8),('dloc',None),('aD',ix['D']),('ours',ix['A']),('v5',ix['v5']),('v8r',ix['v8r'])):
    v=(a[:,i8]-a[:,i7]) if kx is None else a[:,kx]
    print('%-5s post mean %.5f 5/95 %.5f %.5f'%(nm,wm(v),wq(v,.05),wq(v,.95)))
bm=res.sum(1)*tau**2/(SIG**2+k_*tau**2)
fit=wm(P+bm[:,None]); print('fit gains vs EN',dict(zip(('Q2','R1','S1','T1','G1','G2','G3'),fit.round(6))),'\nobs             ',obs.round(6),'\nresid (obs-fit) ',(obs-fit).round(6))
print('posterior-mean gains vs S1: G1 %+.6f G2 %+.6f G3 %+.6f; G3 vs G2 %+.6f (obs +.000004)'%tuple(list(fit[4:]-fit[2])+[fit[6]-fit[5]]))
# resample
ri=np.random.RandomState(9).choice(N,12000,p=Wt)
np.savez(os.environ.get('OUT','post2.npz'),a=a[ri],RR=RR[ri],bm=bm[ri])
print('saved',flush=True)
