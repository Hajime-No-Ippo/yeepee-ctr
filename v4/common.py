"""v4 shared helpers. Imports v3 modules read-only (v3 is never edited)."""
import sys, os
V3 = '/workspace/huawei/v3'
if V3 not in sys.path: sys.path.insert(0, V3)
import numpy as np, polars as pl
from sklearn.metrics import roc_auc_score
from split import J20, DAY, first_events
D4 = os.path.dirname(os.path.abspath(__file__))
HDAY = DAY // 2
# staleness buckets (tm_stale = event ts - cutoff)
BUCKETS = [(0, DAY, '<=1d'), (DAY, 2 * DAY, '1-2d'), (2 * DAY, 10**9, '>2d')]

def bucket_of(stale):
    s = np.asarray(stale); b = np.zeros(len(s), np.int8)
    b[s >= DAY] = 1; b[s >= 2 * DAY] = 2
    return b

_EVAL_SHARE = None
def eval_share():
    """Row share per staleness bucket in eval (from tm_stale of the unlabelled eval rows; ONLY used as validation weights)."""
    global _EVAL_SHARE
    if _EVAL_SHARE is None:
        p = f'{V3}/feat/eval.parquet'
        st = pl.read_parquet(p, columns=['tm_stale'])['tm_stale'].to_numpy()
        _EVAL_SHARE = np.bincount(bucket_of(st), minlength=3) / len(st)
    return _EVAL_SHARE

def report(df, p):
    """df has eid, clicked, tm_stale. Returns dict of global / bucket / eval-weighted / event AUCs."""
    y = df['clicked'].to_numpy(); p = np.asarray(p, np.float64)
    b = bucket_of(df['tm_stale'].to_numpy())
    out = {'auc': roc_auc_score(y, p), 'n': len(y)}
    vs = np.bincount(b, minlength=3) / len(b); es = eval_share()
    for k, (_, _, nm) in enumerate(BUCKETS):
        m = b == k
        out[f'auc_{nm}'] = roc_auc_score(y[m], p[m]) if m.sum() and 0 < y[m].mean() < 1 else None
        out[f'share_{nm}'] = float(vs[k])
    present = vs > 0
    w_b = np.where(present, es / np.where(present, vs, 1), 0)
    # if a bucket is missing in this val set, renormalise eval shares over the buckets present
    w_b = w_b / (es[present].sum())
    out['auc_evalw'] = roc_auc_score(y, p, sample_weight=w_b[b])
    out['evalw_buckets_present'] = int(present.sum())
    ev = pl.DataFrame({'eid': df['eid'], 'y': y, 'p': p}).group_by('eid').agg(pl.col('y').max(), pl.col('p').sum())
    out['auc_event'] = roc_auc_score(ev['y'].to_numpy(), ev['p'].to_numpy())
    return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in out.items()}
