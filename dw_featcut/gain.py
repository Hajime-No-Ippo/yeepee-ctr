import sys; sys.path.insert(0,'.')
from common import *
import pandas as pd
m=src_map()
runs={'nmAs_c23':'v19/ens/nmAs_c23_s0_gain.json','nmAs_c25':'v19/ens/nmAs_c25_s0_gain.json','nmBs_c23':'v19/ens/nmBs_c23_s0_gain.json','nmBs_c25':'v19/ens/nmBs_c25_s0_gain.json','nmBs_c27':'v19/ens/nmBs_c27_s0_gain.json'}
def ld(p):
    d=json.load(open(f'{H}/{p}'))
    return pd.Series({re.sub(r'^coB_','coA_',re.sub(r'^nmB_','nmA_',k)):v for k,v in d.items()})
G=pd.DataFrame({k:ld(p) for k,p in runs.items()}).reindex(REF)
print(G.isna().sum().to_dict())
assert G.notna().all().all()
N=G/G.sum()
N['mean_A']=N[['nmAs_c23','nmAs_c25']].mean(1); N['mean_all']=N.mean(1); N['max_all']=N[list(runs)].max(1)
N['rank_A']=N['mean_A'].rank(ascending=False).astype(int)
N['source_dir']=[m[c] for c in REF]
# legacy split importance from an older v9 LightGBM (377 base cols, 400 trees, lr .05) - proxy only
import lightgbm as lgb
b=lgb.Booster(model_file=f'{H}/v9/ens/trimr_b_c23_s0.txt'); sp=pd.Series(b.feature_importance('split'),index=b.feature_name()); gn=pd.Series(b.feature_importance('gain'),index=b.feature_name())
N['legacy_split']=sp.reindex(REF); N['legacy_gain_share']=(gn/gn.sum()).reindex(REF)
N.to_csv('gain_table.csv')
print(N[['mean_A','mean_all','rank_A']].describe())
print((N.mean_A<1e-5).sum(),(N.mean_A<1e-4).sum(),(N.max_all<1e-4).sum(),(N.max_all<5e-4).sum())
print(N.sort_values('mean_A').head(40)[['mean_A','max_all','source_dir']])
