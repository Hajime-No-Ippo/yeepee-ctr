"""Re-predict v3's saved validation models on v3/feat/val{25,26} keeping the parquet row order (saved valpred .npy rows were
written after an unstable sort, so their row order is unrecoverable). Read-only on v3; output to parallel/."""
import numpy as np, polars as pl, lightgbm as lgb, sys
sys.path.insert(0,'/workspace/huawei/v3')
from metric import global_auc
for cut, mdl in ((25,'full25'),(26,'full26_lr05')):
    m = lgb.Booster(model_file=f'/workspace/huawei/v3/out/{mdl}.txt')
    va = pl.read_parquet(f'/workspace/huawei/v3/feat/val{cut}.parquet')
    p = m.predict(va.select(m.feature_name()).to_numpy().astype(np.float32), num_threads=3)
    print(cut, mdl, 'best_iter', m.best_iteration, 'trees', m.num_trees(), 'auc', round(global_auc(va['clicked'].to_numpy(), p),5), flush=True)
    va.select('rowid').with_columns(pl.Series('p_v3', p)).write_parquet(f'/workspace/huawei/parallel/v3_val{cut}_pred.parquet')
