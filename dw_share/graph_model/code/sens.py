import sys,os; sys.dont_write_bytecode=True
os.chdir('/workspace/huawei/v21fit'); sys.path.insert(0,'/workspace/huawei/v21fit')
import numpy as np
from lib import *
import addon
from addon import sets,ix,n,ks
from binormal import dp,auc,comb as bcomb
from scipy.optimize import brentq
X=nr(np.load('/workspace/huawei/v21graph/out/graph_eval_merged_rank.npy').astype(np.float64))
idx=np.random.RandomState(0).choice(len(EN),200000,replace=False)
rho={k:np.corrcoef(X[idx],sets[k][idx])[0,1] for k in ks}; print('rho EN %.4f A %.4f'%(rho['EN'],rho['A']))
base={'25':.81921,'23':.79903}; offs=[]
for cut,g,w,r in (('25',.00073,.125,.867),('23',.00065,.125,rho['A'])):
    a2=brentq(lambda a_: bcomb(base[cut],a_,r,w)-base[cut]-g,.70,.90); offs.append(a2-base[cut])
off=np.mean(offs); print('offset',off)
P=np.load('post5.npz'); dd,RR=P['d'],P['R']; K=len(dd)
c,g_,q,v8,v5=np.load('pS1.npy')
wS1=np.zeros(n); wS1[0]=1; wS1[1]=-c; wS1[ix['A']]=g_*(1-q); wS1[ix['D']]=g_*q; wS1[ix['v8r']]=v8; wS1[ix['v5']]=v5
aA=auc(dd[:,ix['A']]); rv=np.array([rho[k] for k in ks]); m=n+1
def run(shift,ws_list,seed=5):
    rs=np.random.RandomState(seed); anew=aA+off+shift+rs.normal(0,.0008,K); D2=np.c_[dd,dp(anew)]; R2=np.zeros((K,m,m)); R2[:,:n,:n]=RR
    for t in range(K):
        j=np.minimum(rv+rs.normal(0,1,n)*np.where(rv>.99,.0008,.003),.9999); R2[t,n,:n]=j; R2[t,:n,n]=j; R2[t,n,n]=1
    A_=lambda w: auc((D2@w)/np.sqrt(np.einsum('i,nij,j->n',w,R2,w)))
    w0=np.r_[wS1,0]; b=A_(w0); out={}
    for ws in ws_list:
        g=A_(w0+ws*np.eye(m)[n])-b; out[ws]=(g.mean(),*np.percentile(g,[5,95]),(g>0).mean())
    return anew.mean(),out
print('weight | prior shift | graph public AUC prior | gain mean | 5% | 95% | P>0')
for sh in (0,-.005,.005):
    am,o=run(sh,(-.02,-.045,-.07))
    for ws,v in o.items(): print('%+.3f | %+.3f | %.5f | %+.6f | %+.6f | %+.6f | %.2f'%(ws,sh,am,*v))
