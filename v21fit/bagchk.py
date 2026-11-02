import sys; sys.argv=['x']
exec(open('mkD.py').read().split("cols=")[0])
from scipy.stats import spearmanr
D=np.load('D_dcn.npy')
def bag(spec): return nr(ordered(np.mean([tologit(np.load(f'ens/{t}_A_c27_s{s}.npy')[:,c]) for t,c,s in spec],0),mrow))
cur=[('dcn6',2,0),('dcn6b',2,1),('dcn7',1,0),('dcn4',2,0)]
for nm,spec in {'more snapshots(+ep adjacent)':cur+[('dcn6',1,0),('dcn6b',1,1),('dcn7',2,0),('dcn4',1,0),('dcn4',3,0)],'ep1 only':[('dcn6',0,0),('dcn6b',0,1),('dcn7',0,0),('dcn4',0,0)]}.items():
    x=bag(spec); print(nm,'rho D',round(spearmanr(x[::3],D[::3])[0],5))
for t,c,s in cur: x=nr(ordered(tologit(np.load(f'ens/{t}_A_c27_s{s}.npy')[:,c]),mrow)); print('member',t,c,'rho D',round(spearmanr(x[::3],D[::3])[0],5))
