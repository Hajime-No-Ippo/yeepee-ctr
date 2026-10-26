"""lag-1d: val25 targets (first event/user >= Jan25) with ALL stats frozen at Jan24 (v3feat_ne.build from snapshot)."""
import sys; sys.path.insert(0,'/workspace/huawei/parallel/src_snapshot/v2work'); sys.path.insert(1,'/workspace/huawei/v3')
from v3feat_ne import load, build, log
from split import J20, DAY, first_events
TE, TR, EE, ER, NI, NU = load('/workspace/huawei/v3/data')
T_ev = first_events(TE, J20 + 5 * DAY)
R = build(J20 + 4 * DAY, T_ev, TR.join(T_ev.select('eid'), on='eid', how='semi'), TE, TR, NI, NU)
R.write_parquet('/workspace/huawei/parallel/lag/feat_v3/val25lag.parquet'); log('wrote', R.shape)
