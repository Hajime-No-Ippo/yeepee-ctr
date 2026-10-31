"""report.py: delta table (rolling / C23 / C25 vs public) from score_ref_*.json written by rollval.py score"""
import json, numpy as np
R='/workspace/huawei/v21roll'
PUB={'v12':.818206,'coA':.818270,'coB':.818302,'coBs':.818737}   # p33_w80 probes: v12, v17cooc A, B, v21slate B
pairs=[('v12','coA'),('coA','coB'),('v12','coB'),('coB','coBs'),('coA','coBs'),('v12','coBs')]
keys=['plain3','plain4','plain5','plain6','sm4','sm5','sm6','C23','C25','plain_K4_mean','plain_K4_lin','plain_K4_exp','plain_K3_mean','plain_K3_lin','plain_K3_exp','sm_K3_mean','sm_K3_lin','sm_K3_exp']
J={r:json.load(open(f'{R}/score_ref_{r}.json')) for r in ('v12','coA','coB')}
def val(j,n,k):
    if k in j['res']: return j['res'][k][n]
    g,_,s=k.rpartition('_'); return j['summary'][n][g][s]
def sdv(j,n,k):
    d=j['delta_sd'][n]; return d.get(k)
rows={}
for a,b in pairs:
    j=J[a]; rows[(a,b)]={k:(val(j,b,k)-val(j,a,k),sdv(j,b,k)) for k in keys}
pub=np.array([PUB[b]-PUB[a] for a,b in pairs])
print('pair, public delta:',{f'{a}->{b}':round(PUB[b]-PUB[a],6) for a,b in pairs})
print('%-14s'%'metric'+''.join('%14s'%f'{a}>{b}' for a,b in pairs)+'   corr_w_public  ratio(sum loc/sum pub)  sd(boot)')
for k in keys:
    d=np.array([rows[p][k][0] for p in pairs]); sd=np.mean([rows[p][k][1] or np.nan for p in pairs[:3]+pairs[3:4]])
    print('%-14s'%k+''.join('%+14.5f'%x for x in d)+'   %+.2f  %6.2f   %.5f'%(np.corrcoef(d,pub)[0,1],d.sum()/pub.sum(),sd))
print('\nabs AUCs');
for k in keys: print('%-14s'%k+''.join('%10.5f'%val(J['v12'],n,k) for n in ('v12','coA','coB','coBs')))
print('\nn_slates',J['v12']['n_slates']); print('mix',J['v12']['mix']); print('auc_sd(bootstrap over slates)',{k:{n:round(v,5) for n,v in d.items()} for k,d in J['v12']['auc_sd'].items() if k in('plain3','plain4','plain5','plain6','sm4','sm5','sm6','C23','C25')}['plain6'])
