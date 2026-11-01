import sys; sys.path.insert(0,'/workspace/huawei/v21seq')
import glob
from evalseq import *
from scipy.stats import rankdata,norm
nr=lambda x:(rankdata(x)-1)/(len(x)-1)
c=int(sys.argv[1]);y=yv(c); zl=zbase('A',c); fp=sorted(glob.glob(f'/workspace/huawei/v21seq/ens/ctl6_A_c{c}_s*.npy'))
Ml=np.mean([tologit(np.load(f)[:,-1]) for f in fp],0)
cfg=(('dcn4',2),('dcn6',2),('dcn7',1)) if c==25 else (('dcn4',2),('dcn6',2),('dcn7',1))
Dl=np.mean([tologit(np.load(f'ens/{t}_A_c{c}_s0.npy')[:,e]) for t,e in cfg],0)
w=np.array([.3,.3,.4])
def mix(T): return sum(wi*T(v) for wi,v in zip(w,(zl,Ml,Dl)))
forms={'rank':lambda v:nr(v),'zscore(logit)':lambda v:(v-v.mean())/v.std(),'probit-rank':lambda v:norm.ppf((rankdata(v)-.5)/len(v)),'rank^2(ish) power .5':lambda v:nr(v)**.5,'rank^2':lambda v:nr(v)**2,'prob (sigmoid logit)':lambda v:1/(1+np.exp(-v))}
for k,T in forms.items(): print(c,k,round(AUC(y,mix(T)),5))
