import sys; sys.path.insert(0,'.')
from common import *
import pandas as pd, re
N=pd.read_csv('gain_table.csv',index_col=0); v20=set(json.load(open(f'{H}/v20/out/redundancy.json'))['dropped'])
def fam(c,s):
    if s=='feat_v3':
        for p,f in [('ev_','ev_*'),('hist_','hist_*'),('us_','us_*'),('it_','it_*'),('co_','co_*'),('sv_','sv_*'),('mf_','mf_*')]:
            if c.startswith(p): return f
        return 'v3_other'
    return s
N['family']=[fam(c,s) for c,s in zip(N.index,N.source_dir)]
red=N.redundant_with.notna(); both=[c in v20 for c in N.index]; N['red_both']=red&pd.Series(both,index=N.index)
N['red_any']=red|pd.Series(both,index=N.index)
lowA=N.max_all<1e-4; zero=N.max_all<2e-5
N['tier']='keep'
N.loc[lowA|N.red_any,'tier']='probably_safe'
N.loc[zero|N.red_both,'tier']='safe'
# guard: never auto-cut top-60 gain feature
N.loc[(N.rank_A<=60),'tier']='keep'
N.to_csv('gain_table.csv')
print(N.tier.value_counts())
print(N.groupby(['family','tier']).size().unstack(fill_value=0).assign(total=lambda d:d.sum(1)))
print('gain share of safe',N[N.tier=='safe'].mean_all.sum(),'prob',N[N.tier=='probably_safe'].mean_all.sum())
print(N[N.family=='v3_other'].index.tolist())
print(N[N.tier=='probably_safe'][['family','max_all','redundant_with']].sort_values('family').to_string())
