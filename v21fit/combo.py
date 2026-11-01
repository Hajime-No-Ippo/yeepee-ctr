import sys; sys.path.insert(0,'/workspace/huawei/v21seq')
import glob
from evalseq import *
from scipy.stats import rankdata
nr=lambda x:(rankdata(x)-1)/(len(x)-1)
c=25;y=yv(c); z=nr(zbase('A',c)); fp=sorted(glob.glob(f'/workspace/huawei/v21seq/ens/ctl6_A_c{c}_s*.npy'))
M=nr(np.mean([tologit(np.load(f)[:,-1]) for f in fp],0))
D=np.mean([tologit(np.load(f'ens/{t}_A_c25_s0.npy')[:,e]) for t,e in (('dcn4',2),('dcn6',2),('dcn7',1))],0); Dr=nr(D)
print('D3 AUC',round(AUC(y,Dr),5),'rho M',round(np.corrcoef(Dr,M)[0,1],4))
for wz in (.3,.4):
  for wd in (.2,.3,.4,.5):
    print(wz,wd,round(AUC(y,wz*z+(1-wz-wd)*M+wd*Dr),5))
