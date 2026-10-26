"""v4 step 4: event model + choice model -> FEATURES ONLY for the main binary model.
For each cutoff cv (25, 26, 22 for validation; 27 = final/eval):
  training rows = g_k with C_k < cv, target ts < cv (same leakage rule as protocol.py)
  event model : binary LGBM on one row per event (event-constant features + n_items), label = any click
  choice model: lambdarank on CLICKED events only (all row features); p_share = within-event softmax(s_choice)  [feature only]
  training rows get OOF values (4 folds grouped by user hash); the target table (val g4/g10/g12, or eval) gets full-model values.
Writes feat/stage_{tag}_c{cv}_{name}.parquet with eid, ii, p_event, p_share, p_ev_share."""
import sys, time, argparse
import numpy as np, polars as pl, lightgbm as lgb
import common
from common import D4, report
from features import feature_cols, log
from protocol import train_tables, VAL
SEED = 42; NF = 3; SAMPLE = 0.5  # same event-hash filter as protocol.py --sample 0.5 (salt 0)
EV_P = dict(objective='binary', learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.8,
            bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=SEED, deterministic=True, force_row_wise=True)
CH_P = dict(objective='lambdarank', learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.7,
            bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, lambdarank_truncation_level=20, verbose=-1, seed=SEED,
            deterministic=True, force_row_wise=True)
EV_IT, CH_IT = 250, 200

def samp(df, Cv):
    if Cv is None or SAMPLE >= 1.0: return df  # target tables (val/eval) are never sampled
    return df.filter(((pl.col('eid').cast(pl.UInt64) * 2654435761) % 1000) < int(SAMPLE * 1000))

def ev_cols(allc):
    keep = []
    for c in allc:
        if c in ('eid', 'u', 'ts', 'rowid', 'ii', 'clicked'): continue
        if c.startswith(('us_', 'tm_', 'pd_us_', 'es_')) or c == 'fi_cold' or (c.startswith('ev_') and c.endswith(('_max', '_mean'))):
            keep.append(c)
    return keep

def load_tbl(path, Cv, ecols, rcols):
    sc = pl.read_parquet_schema(path).keys()
    df = pl.read_parquet(path, columns=[c for c in dict.fromkeys(['eid', 'ii', 'u', 'ts', 'clicked'] + ecols + rcols) if c in sc])
    if 'clicked' not in df.columns: df = df.with_columns(pl.lit(0, pl.Int8).alias('clicked'))
    if Cv is not None: df = samp(df.filter(pl.col('ts') < Cv), Cv)
    ev = df.group_by('eid', maintain_order=True).agg([pl.col('u').first(), pl.col('clicked').max().alias('y'),
                                                      pl.len().alias('n_items')] + [pl.col(c).first() for c in ecols])
    return df, ev

def softmax_ev(eid, s):
    d = pl.DataFrame({'eid': eid, 's': s}).with_columns((pl.col('s') - pl.col('s').max().over('eid')).exp().alias('e'))
    return (d['e'] / d.select(pl.col('e').sum().over('eid'))['e']).to_numpy()

def fold_of(u):  # user-grouped folds; u is a string id
    return (pl.Series(u).hash(seed=7).to_numpy() % NF).astype(np.int8)

def run(tag, cv, threads):
    t0 = time.time()
    tgt = (f'{D4}/feat/eval.parquet', None, 'eval') if cv == 27 else (f'{D4}/feat/{VAL[cv]}.parquet', None, VAL[cv])
    tabs = train_tables('g', cv)
    allc = list(pl.read_parquet_schema(tgt[0]).keys())
    probe = pl.DataFrame({c: [] for c in allc})
    rcols = feature_cols(probe, ['it', 'hist', 'us', 'co', 'ev', 'tm', 'sv', 'mf', 'pd', 'fi'])
    ecols = ev_cols(allc)
    evf = ecols + ['n_items']
    # ---- assemble training: events (all) + clicked-event rows
    EV, CR, PARTS = [], [], []
    for path, Cv, name in tabs:
        df, ev = load_tbl(path, Cv, ecols, [])
        EV.append(ev.with_columns(pl.lit(name).alias('tbl')))
        PARTS.append((name, df.select('eid', 'ii', 'u')))
        del df
    EV = pl.concat(EV); EV = EV.with_columns(pl.Series('fold', fold_of(EV['u'])))
    for path, Cv, name in tabs:
        df = pl.read_parquet(path, columns=['eid', 'ii', 'u', 'ts', 'clicked'] + rcols)
        if Cv is not None: df = samp(df.filter(pl.col('ts') < Cv), Cv)
        df = df.filter(pl.col('clicked').max().over('eid') == 1).sort('eid', 'ii')
        CR.append(df.with_columns(pl.lit(name).alias('tbl'))); del df
    CR = pl.concat(CR); CR = CR.with_columns(pl.Series('fold', fold_of(CR['u'])))
    # sort CR by (tbl, eid) so query groups are contiguous
    CR = CR.sort('tbl', 'eid', 'ii')
    log(f'stage cv={cv} train events={len(EV)} clicked-event rows={len(CR)}')
    Xe = EV.select(evf).to_numpy().astype(np.float32); ye = EV['y'].to_numpy(); fe = EV['fold'].to_numpy()
    Xc = CR.select(rcols).to_numpy().astype(np.float32); yc = CR['clicked'].to_numpy(); fc = CR['fold'].to_numpy()
    qkey = (CR['tbl'] + '|' + CR['eid'].cast(pl.Utf8)).to_numpy()
    ep = dict(EV_P, num_threads=threads); cp = dict(CH_P, num_threads=threads)

    def groups(mask):
        k = qkey[mask]; _, idx, cnt = np.unique(k, return_index=True, return_counts=True)
        return cnt[np.argsort(idx)]
    # ---- OOF event model on training events
    oof_e = np.zeros(len(Xe))
    for f in range(NF):
        tr = fe != f
        m = lgb.train(ep, lgb.Dataset(Xe[tr], ye[tr], feature_name=evf), EV_IT)
        oof_e[~tr] = m.predict(Xe[~tr])
    me = lgb.train(ep, lgb.Dataset(Xe, ye, feature_name=evf), EV_IT)
    from sklearn.metrics import roc_auc_score
    oof_auc = roc_auc_score(ye, oof_e)
    log(f'  event model OOF AUC={oof_auc:.4f}')
    # ---- choice model (lambdarank, clicked events only); OOF scores computed for ALL training rows below
    mc_f = []
    for f in range(NF):
        tr = fc != f
        mc_f.append(lgb.train(cp, lgb.Dataset(Xc[tr], yc[tr], group=groups(tr), feature_name=rcols), CH_IT))
    mc = lgb.train(cp, lgb.Dataset(Xc, yc, group=groups(np.ones(len(yc), bool)), feature_name=rcols), CH_IT)
    del Xc, CR
    # ---- write per-table stage features (training tables: OOF by user fold)
    def write(name, path, Cv, oof):
        df = pl.read_parquet(path, columns=['eid', 'ii', 'u', 'ts'] + rcols)
        if Cv is not None: df = samp(df.filter(pl.col('ts') < Cv), Cv)
        _, ev = load_tbl(path, Cv, ecols, [])
        X = df.select(rcols).to_numpy().astype(np.float32)
        Xev = ev.select(evf).to_numpy().astype(np.float32)
        if oof:
            fr = fold_of(df['u']); fv = fold_of(ev['u'])
            s = np.zeros(len(df)); pe = np.zeros(len(ev))
            for f in range(NF):
                mr = fr == f; s[mr] = mc_f[f].predict(X[mr])
            # event OOF: reuse oof_e by (tbl, eid)
            pe = EV.filter(pl.col('tbl') == name).select('eid').with_columns(pl.Series('p', oof_e[(EV['tbl'] == name).to_numpy()]))
            evp = ev.select('eid').join(pe, on='eid', how='left')['p'].to_numpy()
        else:
            s = mc.predict(X); evp = me.predict(Xev)
        out = df.select('eid', 'ii').with_columns(pl.Series('s_choice', s))
        out = out.join(ev.select('eid').with_columns(pl.Series('p_event', evp)), on='eid', how='left')
        out = out.with_columns(pl.Series('p_share', softmax_ev(out['eid'], out['s_choice'].to_numpy())))
        out = out.with_columns((pl.col('p_event') * pl.col('p_share')).alias('p_ev_share')).drop('s_choice')
        out.write_parquet(f'{D4}/feat/stage_{tag}_c{cv}_{name}.parquet')
        return out
    for path, Cv, name in tabs:
        write(name, path, Cv, True)
    out = write(tgt[2], tgt[0], None, False)
    res = {'cv': cv, 'event_oof_auc_train': oof_auc, 'secs': round(time.time() - t0)}
    if cv != 27:
        v = pl.read_parquet(tgt[0], columns=['eid', 'ii', 'clicked', 'tm_stale']).join(out, on=['eid', 'ii'], how='left').sort('eid', 'ii')
        for c in ('p_event', 'p_share', 'p_ev_share'):
            res[c] = report(v, v[c].to_numpy())
        ve = v.group_by('eid').agg(pl.col('clicked').max(), pl.col('p_event').first())
        res['event_model_event_auc'] = roc_auc_score(ve['clicked'].to_numpy(), ve['p_event'].to_numpy())
    log(f'  stage cv={cv} done {res}')
    return res

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('tag'); ap.add_argument('cvs'); ap.add_argument('--threads', type=int, default=4)
    a = ap.parse_args()
    import json, os
    for cv in [int(x) for x in a.cvs.split(',')]:
        r = run(a.tag, cv, a.threads)
        p = f'{D4}/out/stage_{a.tag}.json'
        d = json.load(open(p)) if os.path.exists(p) else {}
        d[str(cv)] = r; json.dump(d, open(p, 'w'), indent=1, default=float)
