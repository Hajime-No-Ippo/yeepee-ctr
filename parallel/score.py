"""Like-for-like scoring of saved val predictions (v3 vs v2work) on identical rows. Read-only on v2work/v3."""
import sys, json, numpy as np, polars as pl
sys.path.insert(0, '/workspace/huawei/v4')
from common import report
from scipy.stats import spearmanr
V2='/workspace/huawei/v2work'; V3='/workspace/huawei/v3'
out={}
for cut, v3pred, v2preds in [(25, 'full25.txt', {'v2work_notime_eft_ne (combo_val25_notime.npy, 300r)':'combo_val25_notime.npy',
                                                        'v2work_notime_x2 (04:26)':'combo_val25_notime_x2.npy',
                                                        'v2work_screen_base_25pct':'scr_base_c25.npy'}),
                             (26, 'full26_lr05.txt', {'v2work_screen_base_25pct (scr_base_c26.npy)':'scr_base_c26.npy'})]:
    # v3 val table: load_table sorts by (fold, eid) -> rows sorted by eid with stable intra-event order
    v3 = pl.read_parquet(f'{V3}/feat/val{cut}.parquet', columns=['rowid','eid','clicked','tm_stale'])
    v3 = v3.join(pl.read_parquet(f'/workspace/huawei/parallel/v3_val{cut}_pred.parquet'), on='rowid', how='left')  # re-predicted (v3_repredict.py)
    assert v3['p_v3'].null_count()==0
    v2 = pl.read_parquet(f'{V2}/feat_v3/val{cut}.parquet', columns=['rowid','eid','clicked','tm_stale'])
    for nm,f in v2preds.items():
        p=np.load(f'{V2}/{f}'); assert len(p)==v2.height, (f,len(p),v2.height); v2=v2.with_columns(pl.Series(nm,p))
    rs3=set(v3['rowid'].to_list()); rs2=set(v2['rowid'].to_list())
    j = v3.join(v2.drop('eid','tm_stale').rename({'clicked':'clicked2'}), on='rowid', how='inner')
    res={'rows_v3':v3.height,'rows_v2work':v2.height,'rows_common':j.height,'only_v3':len(rs3-rs2),'only_v2work':len(rs2-rs3),
         'label_mismatch':int((j['clicked']!=j['clicked2']).sum())}
    res['v3 ('+v3pred+')']=report(j, j['p_v3'].to_numpy())
    for nm in v2preds:
        res[nm]=report(j, j[nm].to_numpy())
        res[nm]['spearman_vs_v3']=round(float(spearmanr(j[nm].to_numpy(), j['p_v3'].to_numpy()).statistic),4)
    out[f'val{cut}']=res
    print(json.dumps({f'val{cut}':res},indent=1),flush=True)
json.dump(out,open('/workspace/huawei/parallel/score_val.json','w'),indent=1)
