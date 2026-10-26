"""Honest validation protocol (v4 step 1).
  - fit on training folds with cutoff < Jan25, EARLY-STOP on val25 -> best_iter B (val25 AUC is ES-biased; logged only)
  - val26 and val22: refit on their own training folds with B scaled by training rows (clip 0.5..1.5), NO early stopping -> report
Training sources:
  v3 : v3 folds f_k (cutoff Jan20+k d, targets first event in [Ck, Ck+1d))  -> val C uses f1..f(C-21)
  g  : v4 folds g_k (cutoff Jan20+k*12h, targets first event in [Ck, Jan27)) -> rows with C_k < Cv and target ts < Cv
Validation tables: val22 = g4, val25 = g10, val26 = g12 (v4 tables; v3 configs use only the v3 columns).
Usage: python protocol.py NAME --src v3|g --groups it,hist,... [--sample 1.0] [--mink 2] [--stage FILE_TAG] [--lr 0.05]"""
import argparse, json, os, time, fcntl
import numpy as np, polars as pl, lightgbm as lgb
import common
from common import J20, DAY, HDAY, D4, V3, report
from features import feature_cols, GROUPS
SEED = 42
PARAMS = dict(objective='binary', metric='auc', learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.7,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=SEED, deterministic=True, force_row_wise=True)
VAL = {22: 'g4', 25: 'g10', 26: 'g12'}
META = ['eid', 'u', 'ii', 'ts', 'clicked', 'tm_stale']

def stage_join(df, tag, name, cv):
    if not tag: return df
    p = f'{D4}/feat/stage_{tag}_c{cv}_{name}.parquet'
    return df.join(pl.read_parquet(p), on=['eid', 'ii'], how='left')

def train_tables(src, Cv_day, mink=2, maxk=None):
    Cv = J20 + (Cv_day - 20) * DAY
    if src == 'v3':
        return [(f'{V3}/feat/f{k}.parquet', None, f'f{k}') for k in range(1, Cv_day - 20)]
    ks = [k for k in range(mink, 14) if J20 + k * HDAY < Cv]
    return [(f'{D4}/feat/g{k}.parquet', Cv, f'g{k}') for k in ks]

def load(tables, feats, sample, tag=None, salt=0, cv=None):
    Xs, metas = [], []
    for path, Cv, name in tables:
        cols = pl.read_parquet_schema(path).keys()
        df = pl.read_parquet(path, columns=[c for c in dict.fromkeys(META + feats) if c in cols])
        if Cv is not None: df = df.filter(pl.col('ts') < Cv)
        if sample < 1.0:
            df = df.filter(((pl.col('eid').cast(pl.UInt64) * 2654435761 + salt) % 1000) < int(sample * 1000))
        df = stage_join(df, tag, name, cv)
        df = df.sort('eid', 'ii')
        Xs.append(df.select(feats).to_numpy().astype(np.float32)); metas.append(df.select(META))
    return np.concatenate(Xs), pl.concat(metas)

def run(name, src, groups, sample=1.0, mink=2, tag=None, lr=0.05, threads=4, drop=(), save_model=False, extra_feats=()):
    t0 = time.time()
    v25 = f'{D4}/feat/{VAL[25]}.parquet'
    allc = list(pl.read_parquet_schema(v25).keys())
    probe = pl.DataFrame({c: [] for c in allc})
    feats = [f for f in feature_cols(probe, groups) if not any(f.startswith(d) for d in drop)] + list(extra_feats)
    if src == 'v3':  # v3 baseline: exactly the v3 feature set
        v3c = pl.read_parquet_schema(f'{V3}/feat/f1.parquet').keys()
        feats = [f for f in feats if f in v3c]
    p = dict(PARAMS, learning_rate=lr, num_threads=threads)
    res = dict(name=name, src=src, groups=groups, sample=sample, mink=mink, stage=tag, lr=lr, n_feats=len(feats))
    # --- val25: early stopping
    Xtr, mtr = load(train_tables(src, 25, mink), feats, sample, tag, cv=25)
    Xva, mva = load([(v25, None, 'g10')], feats, 1.0, tag, cv=25)
    dtr = lgb.Dataset(Xtr, mtr['clicked'].to_numpy(), feature_name=feats, free_raw_data=True)
    dva = lgb.Dataset(Xva, mva['clicked'].to_numpy(), reference=dtr)
    m = lgb.train(p, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    B = m.best_iteration; n25 = len(Xtr)
    res['best_iter25'] = B; res['n_train25'] = n25
    res['val25'] = report(mva, m.predict(Xva, num_iteration=B))
    g = m.feature_importance('gain'); tot = g.sum()
    res['top15'] = [(feats[i], round(float(g[i] / tot), 4)) for i in np.argsort(-g)[:15]]
    del dtr, dva, Xtr
    # --- val26 / val22: fixed scaled iterations, report only
    for cv in (26, 22):
        Xtr, mtr = load(train_tables(src, cv, mink), feats, sample, tag, cv=cv)
        it = int(round(B * float(np.clip(len(Xtr) / n25, 0.5, 1.5))))
        mm = lgb.train(p, lgb.Dataset(Xtr, mtr['clicked'].to_numpy(), feature_name=feats), it)
        Xv, mv = load([(f'{D4}/feat/{VAL[cv]}.parquet', None, VAL[cv])], feats, 1.0, tag, cv=cv)
        pv = mm.predict(Xv)
        res[f'val{cv}'] = report(mv, pv); res[f'iters{cv}'] = it; res[f'n_train{cv}'] = len(Xtr)
        if save_model and cv == 26:
            mm.save_model(f'{D4}/out/{name}_m26.txt'); np.save(f'{D4}/out/{name}_val26pred.npy', pv)
        del Xtr
    res['secs'] = round(time.time() - t0)
    return res

def record(res, step):
    p = f'{D4}/out/ablation.json'
    with open(p + '.lock', 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        d = json.load(open(p)) if os.path.exists(p) else {}
        d.setdefault(step, {})[res['name']] = res
        json.dump(d, open(p, 'w'), indent=1)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('name'); ap.add_argument('--step', default='step1')
    ap.add_argument('--src', default='g'); ap.add_argument('--groups', default=','.join(GROUPS))
    ap.add_argument('--drop', default=''); ap.add_argument('--sample', type=float, default=1.0)
    ap.add_argument('--mink', type=int, default=2); ap.add_argument('--stage', default=None)
    ap.add_argument('--extra', default=''); ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--threads', type=int, default=4); ap.add_argument('--save', action='store_true')
    a = ap.parse_args()
    r = run(a.name, a.src, a.groups.split(','), a.sample, a.mink, a.stage, a.lr, a.threads,
            [d for d in a.drop.split(',') if d], a.save, [e for e in a.extra.split(',') if e])
    print(json.dumps(r)); record(r, a.step)
