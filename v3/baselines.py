"""Single-score baselines on the local validation sets (val25 / val26), incl. a faithful port of starter_solution.py
fitted only on events < cutoff. Appends to out/baselines.json."""
import json, os, sys
import numpy as np, polars as pl
from features import load
from split import C_MAIN, C_SECOND, split
from metric import global_auc
D = os.path.dirname(os.path.abspath(__file__))
TE, TR, EE, ER, NI, NU = load(f'{D}/data')
K = 50; CFW = 25.0
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
res = {}
for tag, C in ((25, C_MAIN), (26, C_SECOND)):
    fe, fr, ve, vr = split(TE, TR, C)
    P = fr.join(fe.select('eid', 'u'), on='eid')
    g = P['clicked'].mean()
    ic = P.group_by('ii').agg(((pl.col('clicked').sum() + K * g) / (pl.len() + K)).alias('ictr'))
    uc = P.group_by('u').agg(((pl.col('clicked').sum() + K * g) / (pl.len() + K)).alias('uctr'))
    inter = pl.concat([fe.select('u', 'h').explode('h').drop_nulls().rename({'h': 'ii'}).with_columns(pl.col('ii').cast(pl.Int32)),
                       P.filter(pl.col('clicked') == 1).select('u', pl.col('ii').cast(pl.Int32))]).unique()
    pop = inter.group_by('ii').agg(pl.len().alias('pop'))
    pairs = (inter.join(inter, on='u', suffix='_b').filter(pl.col('ii') != pl.col('ii_b')).group_by('ii', 'ii_b').agg(pl.len().alias('c'))
             .join(pop, on='ii').join(pop.rename({'ii': 'ii_b', 'pop': 'pop_b'}), on='ii_b')
             .with_columns((pl.col('c') / (pl.col('pop') * pl.col('pop_b')).sqrt()).alias('cs')).select('ii', 'ii_b', 'cs'))
    R = vr.join(ve.select('eid', 'u', 'h'), on='eid').with_row_index('r')
    PR = R.select('r', pl.col('ii').cast(pl.Int32), 'h').explode('h').drop_nulls('h').rename({'h': 'ii_b'}).with_columns(pl.col('ii_b').cast(pl.Int32))
    cf = PR.join(pairs, on=['ii', 'ii_b'], how='left').group_by('r').agg(pl.col('cs').fill_null(0).sum().alias('cf'))
    R = (R.join(ic, on='ii', how='left').join(uc, on='u', how='left').join(cf, on='r', how='left')
          .with_columns(pl.col('ictr').fill_null(g), pl.col('uctr').fill_null(g), pl.col('cf').fill_null(0)).sort('r'))
    y = R['clicked'].to_numpy()
    starter = lg(R['ictr'].to_numpy()) + lg(R['uctr'].to_numpy()) - lg(g) + CFW * R['cf'].to_numpy()
    F = pl.read_parquet(f'{D}/feat/val{tag}.parquet')
    yf = F['clicked'].to_numpy()
    res[tag] = dict(starter=global_auc(y, starter), starter_item_ctr_k50=global_auc(y, R['ictr'].to_numpy()),
                    item_ctr_smoothed=global_auc(yf, F['it_ctr'].to_numpy()), item_impressions=global_auc(yf, F['it_imp'].to_numpy()),
                    cand_in_history=global_auc(yf, F['hist_in'].to_numpy()), n_rows=len(y), n_events=ve.height, ctr=float(y.mean()))
    print(tag, res[tag], flush=True)
json.dump(res, open(f'{D}/out/baselines.json', 'w'), indent=1)
