import polars as pl, numpy as np, sys
from scipy.stats import ks_2samp
d = sys.argv[1]
F = {k: pl.read_parquet(f'{d}/{k}.parquet') for k in ['f3','f4','f5','f6','eval']}
tr = pl.concat([F[k] for k in ['f3','f4','f5','f6']]); ev = F['eval']
print(f"{'feat':12s} {'tr_mean':>9s} {'ev_mean':>9s} {'tr_std':>8s} {'ev_std':>8s} {'KS':>6s}  per-fold means f3..f6")
for c in [c for c in tr.columns if c != 'rowid']:
    a = tr[c].to_numpy().astype(float); b = ev[c].to_numpy().astype(float)
    rs = np.random.default_rng(0); ks = ks_2samp(rs.choice(a, 200000), rs.choice(b, 200000)).statistic
    pm = ' '.join(f"{F[k][c].mean():.3f}" for k in ['f3','f4','f5','f6'])
    print(f"{c:12s} {a.mean():9.4f} {b.mean():9.4f} {a.std():8.4f} {b.std():8.4f} {ks:6.3f}  {pm}")
