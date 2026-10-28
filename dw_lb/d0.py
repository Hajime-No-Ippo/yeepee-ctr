import sys; sys.path.insert(0,'/workspace/huawei/dw_lb')
from lib import *
import dwcommon as D, pickle, time
t=time.time(); d=D.Data(); print('data',time.time()-t, d.NE, d.NU, d.NI)
ec=d.Y.max(1)  # event has click
print('event click rate',ec.mean())
# user heterogeneity: split-half reliability
u=d.EU; order=np.lexsort((d.ETS,u)); 
us=u[order]; bounds=np.r_[0,np.flatnonzero(np.diff(us))+1,len(us)]
n=np.diff(bounds); print('events/user mean',n.mean(),'median',np.median(n), 'min',n.min(),'max',n.max())
ecs=ec[order]; e1=[];e2=[];ns=[]
for a,b in zip(bounds[:-1],bounds[1:]):
    if b-a>=8:
        h=(b-a)//2; e1.append(ecs[a:a+h].mean()); e2.append(ecs[b-h:b].mean()); ns.append(h)
e1=np.array(e1);e2=np.array(e2); print('users>=8 ev',len(e1),'corr halves',np.corrcoef(e1,e2)[0,1],'mean ns',np.mean(ns))
# expected corr under homogeneity is 0; also compare to simulated binomial var
print('var of half-rate',e1.var(),'binomial var',np.mean(ec.mean()*(1-ec.mean())/np.array(ns)))
# by hour of day, by time since last event
ts=d.ETS; hr=((ts%DAY)//3600)
print('click rate by hour',[round(ec[hr==h].mean(),3) for h in range(24)])
gap=np.full(len(ts),np.nan); o=order; g=np.diff(ts[o]); same=(us[1:]==us[:-1]); tmp=np.full(len(o),np.nan); tmp[1:]=np.where(same,g,np.nan); gap[o]=tmp
for lo,hi in [(0,1800),(1800,3600*3),(3*3600,6*3600),(6*3600,12*3600),(12*3600,24*3600),(24*3600,48*3600),(48*3600,1e9)]:
    m=(gap>=lo)&(gap<hi); print('gap',lo/3600,hi/3600,m.sum(),round(ec[m].mean(),4))
