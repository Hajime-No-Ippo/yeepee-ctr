"""Leakage checks for the sv (SVD item-embedding / impression-contrast) features.
1) asserts: every history click feeding any click log has ts < C (train folds, val, eval)
2) rebuild f1..f4 + val25 with sv matrix variants and retrain at cutoff 25:
   notgt = drop target events' own histories from the matrix; lag1d = matrix from clicks/impressions ts < C-1d
Usage: python leakcheck.py [notgt|lag1d ...]"""
import os, sys, json, time
import numpy as np, polars as pl
import train as T
from features import load, build, log, feature_cols, GROUPS, _hist_long
from split import J20, DAY, first_events
D = os.path.dirname(os.path.abspath(__file__))
TE, TR, EE, ER, NI, NU = load(f'{D}/data')
res = {}
# --- raw checks: history timestamps vs own event time and vs cutoff
for nm, E in (('train', TE), ('eval', EE)):
    x = E.select('eid', 'ts', 'h_ts').explode('h_ts').drop_nulls('h_ts')
    res[f'{nm}_hist_after_event'] = int((x['h_ts'] >= x['ts']).sum())
for C, nm in ((J20 + 5 * DAY, 'val25'), (J20 + 6 * DAY, 'val26')):
    T_ev = first_events(TE, C)
    h = _hist_long(T_ev)
    res[f'{nm}_target_hist_clicks_ge_C'] = int((h['ts'] >= C).sum())
    res[f'{nm}_target_min_ts_ge_C'] = bool(T_ev['ts'].min() >= C)
res['eval_min_event_ts_ge_Jan27'] = bool(EE['ts'].min() >= J20 + 7 * DAY)
res['train_max_event_ts_lt_Jan27'] = bool(TE['ts'].max() < J20 + 7 * DAY)
res['eval_hist_clicks_ge_Jan27'] = int((_hist_long(EE)['ts'] >= J20 + 7 * DAY).sum())
print(json.dumps(res, indent=1), flush=True)
feats = None
for mode in (sys.argv[1:] or ['notgt', 'lag1d']):
    od = f'{D}/feat_leak/{mode}'; os.makedirs(od, exist_ok=True)
    t0 = time.time()
    for name, C, W in [(f'f{k}', J20 + k * DAY, DAY) for k in range(1, 5)] + [('val25', J20 + 5 * DAY, None)]:
        out = f'{od}/{name}.parquet'
        if os.path.exists(out): continue
        T_ev = first_events(TE, C, W)
        build(C, T_ev, TR.join(T_ev.select('eid'), on='eid', how='semi'), TE, TR, NI, NU, sv_mode=mode).write_parquet(out)
    tr = pl.concat([pl.read_parquet(f'{od}/f{k}.parquet').with_columns(pl.lit(k, pl.Int16).alias('fold')) for k in range(1, 5)], how='diagonal_relaxed').sort('fold', 'eid')
    va = pl.read_parquet(f'{od}/val25.parquet').with_columns(pl.lit(99, pl.Int16).alias('fold'))
    fc = feature_cols(tr, GROUPS)
    m, best, auc = T.fit(tr, va, fc, 'binary', 0.08, int(os.environ.get('THREADS', 4)))
    svf = [c for c in fc if c.startswith('sv_')]
    from metric import global_auc
    single = {c: round(global_auc(va['clicked'].to_numpy(), va[c].fill_null(-9).to_numpy()), 4) for c in ('sv_k64_hcos', 'sv_k64_con05', 'sv_k32_ucos')}
    r = dict(mode=mode, val25_auc=auc, best_iter=best, secs=round(time.time() - t0), single=single)
    print(json.dumps(r), flush=True)
    with open(f'{D}/out/leakcheck.jsonl', 'a') as f: f.write(json.dumps(r) + '\n')
