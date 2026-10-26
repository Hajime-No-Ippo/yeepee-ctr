"""v6 final blend: mean logits of base(2 seeds,400r) and slow(1 seed,500r), feat_lr2 extra. Also prints val AUCs."""
import glob, os, sys, numpy as np, polars as pl
from cl import matrix
from sklearn.metrics import roc_auc_score
COL = {'base': 1, 'slow': 2}; NS = {'base': 2, 'slow': 1}
PFX = os.environ.get('V6PFX', 'v6'); EX = ('feat_lr2',)
def blend(cut):
    c = {}
    for t, ci in COL.items():
        fs = sorted(glob.glob(f'ens/{PFX}{t}_c{cut}_s*.npy')); assert len(fs) == NS[t], (t, fs)
        c[t] = np.mean([np.load(f)[:, ci] for f in fs], 0)
    return c, np.mean(list(c.values()), 0)
if 'val' in sys.argv:
    for cut in (25, 26):
        _, m, _ = matrix(f'val{cut}', 1.0, extra=EX); y = m['clicked'].to_numpy()
        c, z = blend(cut)
        print(cut, {k: round(roc_auc_score(y, v), 5) for k, v in c.items()}, 'blend', round(roc_auc_score(y, z), 5))
        # v4_lite like-for-like ref (same row order expected? check via base matrix)
        _, m0, _ = matrix(f'val{cut}', 1.0); y0 = m0['clicked'].to_numpy()
        try:
            b = np.mean([np.load(f'ens/base_c{cut}_s{s}.npy')[:, 1] for s in (0, 1)], 0); sl = np.load(f'ens/slow_c{cut}_s0.npy')[:, 2]
            print('  v4_lite', round(roc_auc_score(y0, (b + sl) / 2), 5), 'same rows', bool((m0['rowid'].to_numpy() == m['rowid'].to_numpy()).all()))
        except Exception as e: print('  v4_lite ref err', e)
if 'eval' in sys.argv:
    _, m, _ = matrix('eval', 1.0, extra=EX)
    c, z = blend(27); assert len(z) == m.height
    p = 1 / (1 + np.exp(-z)); print('corr base/slow', round(float(np.corrcoef(c['base'], c['slow'])[0, 1]), 4))
    pl.DataFrame({'rowid': m['rowid'], 'p': p}).write_parquet(os.environ.get('PRED_OUT', 'eval_pred_v6.parquet')); print('wrote', m.height)
