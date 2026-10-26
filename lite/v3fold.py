"""v3 features (v2work/v3feat_ne.py = v3/features.py with global/item stats from train data only) for all fold specs.
Usage: python v3fold.py NAME... -> v2work/feat_v3/NAME.parquet
  f1..f6: first event/user in [Ck, Ck+1d); val25/val26: [C, end of train); eval: Jan27 cutoff
  e1..e4: [Ck, Jan25) (eval-like staleness, for cutoff-25 validation); E1..E6: [Ck, Jan27) (eval-like, final fit)"""
import sys, os
sys.path.insert(0, '/workspace/huawei/lite'); sys.path.insert(1, '/workspace/huawei/lite/v3')
from v3feat_ne import load, build, log
from split import J20, DAY, first_events
TE, TR, EE, ER, NI, NU = load('/workspace/huawei/lite/v3/data')
OUT = '/workspace/huawei/lite/feat_v3'; os.makedirs(OUT, exist_ok=True)
SPEC = {**{f'f{k}': (J20 + k * DAY, DAY) for k in range(1, 7)}, 'val25': (J20 + 5 * DAY, None), 'val26': (J20 + 6 * DAY, None),
        **{f'e{k}': (J20 + k * DAY, (5 - k) * DAY) for k in range(1, 5)}, **{f'E{k}': (J20 + k * DAY, (7 - k) * DAY) for k in range(1, 7)}, **{f'd{k}': (J20 + k * DAY, (6 - k) * DAY) for k in range(1, 6)}}
for n in sys.argv[1:]:
    if n == 'eval': R = build(J20 + 7 * DAY, EE, ER, TE, TR, NI, NU)
    else:
        C, W = SPEC[n]; T_ev = first_events(TE, C, W)
        R = build(C, T_ev, TR.join(T_ev.select('eid'), on='eid', how='semi'), TE, TR, NI, NU)
    R.write_parquet(f'{OUT}/{n}.parquet'); log('wrote', n, R.shape)
