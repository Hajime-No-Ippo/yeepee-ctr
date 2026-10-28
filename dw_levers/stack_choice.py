import sys, json, numpy as np, lightgbm as lgb
sys.path.insert(0,'/workspace/huawei/dw_levers')
from choice import *
from common import meta, z12, A
from sklearn.model_selection import GroupKFold
cut=int(sys.argv[1]); cfg=json.loads(sys.argv[2]); tag=sys.argv[3]
D=Data(); v=D.val(cut); v['col']=align(D,v); z=z12(cut); y=v['y']
import os; f=f'cache/choice_{tag}_c{cut}.npz'
if os.path.exists(f): d=np.load(f); lp,on=d['lp'],d['on']
else:
    mdl=fit_choice(D,Jd(cut),cfg,val=v); lp,on=score(mdl,D,v['e'][::10]); lp=to_val(lp,v['col']).ravel(); on=on; np.savez(f,lp=lp,on=on)
n=len(y)//10; ev=np.repeat(np.arange(n),10)
lz=np.log(z/(1-z)) if z.max()<=1 else z
if z.max()>1: print('z not prob?',z.min(),z.max())
def slate(a): m=a.reshape(n,10); return (m-m.mean(1,keepdims=True)).ravel(), (np.argsort(np.argsort(-m,1),1)).ravel()
Z=np.c_[lz,*slate(lz),np.repeat(np.log(np.exp(lz.reshape(n,10)).sum(1)),10)]
X1=np.c_[Z,lp,*slate(lp),np.repeat(on,10)]
users=v['user']; res={}
def cv(X,name):
    out=np.zeros(len(y))
    for seed in (0,1):
        for tr,te in GroupKFold(5).split(X,y,users):
            p=dict(objective='binary',learning_rate=0.02,num_leaves=8,min_data_in_leaf=300,feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10,seed=seed,verbose=-1,num_threads=4)
            m=lgb.train(p,lgb.Dataset(X[tr],y[tr],init_score=lz[tr]),150)
            out[te]+=m.predict(X[te],raw_score=True)+lz[te]
    out/=2; res[name]=out; return out
base=A(y,lz)
for name,X in (('ctrl',Z),('stack',X1)):
    o=cv(X,name); print(name,'AUC',A(y,o),'delta',A(y,o)-base,flush=True)
# paired event bootstrap of stack-ctrl
rng=np.random.default_rng(0); yy=y.reshape(n,10); a=res['stack'].reshape(n,10); b=res['ctrl'].reshape(n,10); ds=[]
for _ in range(100):
    i=rng.integers(0,n,n); ds.append(A(yy[i].ravel(),a[i].ravel())-A(yy[i].ravel(),b[i].ravel()))
print('stack-ctrl',np.mean(ds),np.percentile(ds,[2.5,97.5]))
