import sys, numpy as np, polars as pl
for new, old in zip(sys.argv[1::2], sys.argv[2::2]):
    a = pl.read_parquet(new); b = pl.read_parquet(old)
    assert a.columns == b.columns, (set(a.columns) ^ set(b.columns)); assert a['rowid'].equals(b['rowid'])
    bad = []; mx = 0
    for c in a.columns:
        if a[c].dtype.is_numeric():
            x = a[c].cast(pl.Float64).to_numpy(); y = b[c].cast(pl.Float64).to_numpy()
            nx, ny = np.isnan(x), np.isnan(y)
            if (nx != ny).any(): bad.append((c, 'nan', int((nx != ny).sum()))); continue
            d = np.abs(x[~nx] - y[~ny]); r = d / (np.abs(y[~ny]) + 1e-6)
            if len(d) and r.max() > 1e-4: bad.append((c, float(r.max()), int((r > 1e-4).sum())))
            mx = max(mx, float(r.max()) if len(r) else 0)
    print(new, 'cols', len(a.columns), 'max rel diff', mx, 'bad', [b[0] for b in bad])
