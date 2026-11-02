import sys; sys.path.insert(0,'../evclick')
from evlib import *
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
B='/workspace/huawei/cb_blend/'
for n in ['f1','f4','eval']:
    d,_=cols(n,[]); t=d['tm_stale'].to_numpy(); print(n,'tm_stale q',np.percentile(t,[10,25,50,67,75,90,99]).round(0),'frac>1d %.3f'%(t>86400).mean())
for c in (23,25,26):
    o=np.load(B+f'oof_c{c}.npz',allow_pickle=True); y=o['y']; rid=o['rowid']
    d,_=cols(f'val{c}',[]); mp=dict(zip(d['rowid'].to_numpy(),d['tm_stale'].to_numpy())); t=np.array([mp[i] for i in rid])
    s=rankdata(o['raw_M'])/len(y)
    print(c,'frac>1d %.3f'%(t>86400).mean(),'AUC all %.4f'%roc_auc_score(y,s),'fresh %.4f'%roc_auc_score(y[t<=86400],s[t<=86400]),'stale %.4f'%roc_auc_score(y[t>86400],s[t>86400]) if (t>86400).any() else '')
