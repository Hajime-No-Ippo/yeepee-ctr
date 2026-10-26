"""LightGBM click model on cached feature tables.

Reusable API:
  load_table(names)            -> concatenated feature frame (sorted by fold, eid)
  fit(tr, va, feats, obj, ...) -> (booster, best_iter, val_auc)
CLI:
  python train.py --cutoff 25 [--groups it,hist,us,co,ev,tm,mf] [--obj binary|lambdarank] [--lr 0.05] [--sample 1.0]
Training rows for cutoff Jan25 = folds f1..f4 (all labels < Jan25); for Jan26 = f1..f5. Validation = val25 / val26.
"""
import argparse, json, os, time
import numpy as np
import polars as pl
import lightgbm as lgb
from features import feature_cols, GROUPS
from metric import global_auc
D = os.path.dirname(os.path.abspath(__file__))
SEED = 42
BASE_PARAMS = dict(learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.7, bagging_fraction=0.8,
                   bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=SEED, deterministic=True, force_row_wise=True)
FOLDS = {25: ['f1', 'f2', 'f3', 'f4'], 26: ['f1', 'f2', 'f3', 'f4', 'f5'], 27: ['f1', 'f2', 'f3', 'f4', 'f5', 'f6']}

def load_table(names, sample=1.0):
    dfs = []
    for k, n in enumerate(names):
        df = pl.read_parquet(f'{D}/feat/{n}.parquet').with_columns(pl.lit(k, pl.Int16).alias('fold'))
        if sample < 1.0:  # sample whole events
            ev = df.select('eid').unique().sort('eid').sample(fraction=sample, seed=SEED)
            df = df.join(ev, on='eid', how='semi')
        dfs.append(df)
    return pl.concat(dfs, how='diagonal_relaxed').sort('fold', 'eid')

def _ds(df, feats, obj, ref=None):
    X = df.select(feats).to_numpy().astype(np.float32); y = df['clicked'].to_numpy()
    kw = {}
    if obj == 'lambdarank':
        kw['group'] = df.group_by(['fold', 'eid'], maintain_order=True).len()['len'].to_numpy()
    return lgb.Dataset(X, y, feature_name=feats, reference=ref, free_raw_data=False, **kw), X, y

def fit(tr, va, feats, obj='binary', lr=0.05, threads=6, max_rounds=3000, es=100, params=None):
    p = dict(BASE_PARAMS, objective=obj, learning_rate=lr, num_threads=threads, metric='auc')
    if obj == 'lambdarank':
        p.update(lambdarank_truncation_level=10)
    if params: p.update(params)
    dtr, _, _ = _ds(tr, feats, obj)
    if va is None:
        m = lgb.train(p, dtr, num_boost_round=max_rounds)
        return m, max_rounds, None
    dva, Xva, yva = _ds(va, feats, obj, ref=dtr)
    m = lgb.train(p, dtr, num_boost_round=max_rounds, valid_sets=[dva], valid_names=['val'],
                  callbacks=[lgb.early_stopping(es, verbose=False), lgb.log_evaluation(200)])
    best = m.best_iteration
    auc = global_auc(yva, m.predict(Xva, num_iteration=best))   # recompute global AUC explicitly
    return m, best, auc

def importance(m, top=15):
    g = m.feature_importance('gain'); names = m.feature_name(); tot = g.sum()
    return [(names[i], float(g[i] / tot)) for i in np.argsort(-g)[:top]]

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cutoff', type=int, default=25)
    ap.add_argument('--groups', default=','.join(GROUPS))
    ap.add_argument('--drop', default='')
    ap.add_argument('--obj', default='binary')
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--sample', type=float, default=1.0)
    ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--save', default='')
    a = ap.parse_args()
    t0 = time.time()
    tr = load_table(FOLDS[a.cutoff], a.sample); va = load_table([f'val{a.cutoff}'])
    feats = feature_cols(tr, a.groups.split(','))
    drops = [d for d in a.drop.split(',') if d]
    feats = [f for f in feats if not any(f.startswith(d) for d in drops)]
    m, best, auc = fit(tr, va, feats, a.obj, a.lr, a.threads)
    res = dict(cutoff=a.cutoff, groups=a.groups, drop=a.drop, obj=a.obj, lr=a.lr, sample=a.sample, n_feats=len(feats),
               best_iter=best, val_auc=auc, secs=round(time.time() - t0), top15=importance(m))
    print(json.dumps(res))
    with open(f'{D}/out/results.jsonl', 'a') as f: f.write(json.dumps(res) + '\n')
    if a.save:
        m.save_model(f'{D}/out/{a.save}.txt')
        np.save(f'{D}/out/{a.save}_valpred.npy', m.predict(va.select(feats).to_numpy().astype(np.float32), num_iteration=best))
