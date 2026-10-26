"""Build v4 feature tables.  python build_feats_v4.py [names...|all] [--force]
  g{k}: cutoff C = Jan20 + k*12h (k=2..13), targets = each user's first event in [C, Jan27)  (eval-like staleness 0-6d)
        val22 = g4, val25 = g10, val26 = g12 (training use filters target ts < validation cutoff)
  eval: cutoff Jan27, targets = evaluation events"""
import os, sys, time
import common
from common import J20, DAY, HDAY, D4, first_events
from features import load, log
from features_v4 import build
TE, TR, EE, ER, NI, NU = load(f'{common.V3}/data')
names = [a for a in sys.argv[1:] if not a.startswith('--')] or ['all']
if names == ['all']: names = [f'g{k}' for k in range(2, 14)] + ['eval']
t0 = time.time()
for nm in names:
    out = f'{D4}/feat/{nm}.parquet'
    if os.path.exists(out) and '--force' not in sys.argv: log('skip', nm); continue
    if nm == 'eval':
        R = build(J20 + 7 * DAY, EE, ER, TE, TR, NI, NU)
    else:
        C = J20 + int(nm[1:]) * HDAY
        T_ev = first_events(TE, C)
        R = build(C, T_ev, TR.join(T_ev.select('eid'), on='eid', how='semi'), TE, TR, NI, NU)
    R.write_parquet(out)
log(f'v4 feature build total {time.time()-t0:.0f}s')
