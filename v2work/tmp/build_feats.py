"""Build & cache feature tables. Usage: python build_feats.py [val|final|all]
  folds f1..f6 : cutoff Ck=Jan20+k d, targets = first event/user in [Ck, Ck+1d)   (model-training rows)
  val25 / val26: cutoff Jan25 / Jan26, targets = first event/user at/after cutoff (to end of train)
  eval         : cutoff Jan27, targets = evaluation events
"""
import os, sys, time
import polars as pl
from features import load, build, log
from split import J20, DAY, first_events
D = os.path.dirname(os.path.abspath(__file__))
TE, TR, EE, ER, NI, NU = load(f'{D}/data')
mode = sys.argv[1] if len(sys.argv) > 1 else 'all'
specs = []
if mode in ('val', 'all'):
    specs += [(f'f{k}', J20 + k * DAY, DAY) for k in range(1, 6)] + [('val25', J20 + 5 * DAY, None), ('val26', J20 + 6 * DAY, None)]
if mode in ('final', 'all'):
    specs += [s for s in [('f6', J20 + 6 * DAY, DAY), ('eval', J20 + 7 * DAY, 'eval')] if s[0] not in [x[0] for x in specs]]
t0 = time.time()
for name, C, W in specs:
    out = f'{D}/feat/{name}.parquet'
    if os.path.exists(out) and '--force' not in sys.argv:
        log('skip', name); continue
    if W == 'eval':
        R = build(C, EE, ER, TE, TR, NI, NU)
    else:
        T_ev = first_events(TE, C, W)
        R = build(C, T_ev, TR.join(T_ev.select('eid'), on='eid', how='semi'), TE, TR, NI, NU)
    R.write_parquet(out)
log(f'feature build total {time.time()-t0:.0f}s')
