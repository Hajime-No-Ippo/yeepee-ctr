import sys; sys.path.insert(0,'.')
from common import *
import pandas as pd, collections
N=pd.read_csv('gain_table.csv',index_col=0); C=json.load(open('corr_out.json')); v20=json.load(open(f'{H}/v20/out/redundancy.json'))
gain=N.mean_all
nan=pd.Series(C['nan_share']); nun=pd.Series(C['nunique']); top=pd.Series(C['top_value_share'])
# greedy redundancy: pair if |Spearman|>0.98 (Pearson alone ignored when rank corr lower; require Spearman, since GBDT is rank-invariant)
adj=collections.defaultdict(dict)
for a,b,p,s in C['pairs']:
    if s is not None and abs(s)>0.98: adj[a][b]=s; adj[b][a]=s
dropped={}; kept=[]
for c in gain.sort_values(ascending=False).index:
    if c in dropped: continue
    kept.append(c)
    for d,s in adj[c].items():
        if d not in dropped and d not in kept: dropped[d]=(c,round(abs(s),4))
mine=set(dropped); theirs=set(v20['dropped'])
print('mine',len(mine),'v20',len(theirs),'both',len(mine&theirs),'only mine',len(mine-theirs),'only v20',len(theirs-mine))
pd.Series({d:f'{k} ({s})' for d,(k,s) in dropped.items()}).to_csv('redundant_dropped.csv')
json.dump({d:{'kept_in_favour_of':k,'abs_spearman':s} for d,(k,s) in dropped.items()},open('redundant_dropped.json','w'),indent=1)
print('const-like (top value share>0.999):',list(top[top>0.999].index)); print('nan>0.99:',list(nan[nan>0.99].index)); print('nunique<=1:',list(nun[nun<=1].index))
N['nan_share']=nan; N['top_value_share']=top; N['redundant_with']=pd.Series({d:k for d,(k,s) in dropped.items()}); N['redundant_rho']=pd.Series({d:s for d,(k,s) in dropped.items()})
N.to_csv('gain_table.csv')
