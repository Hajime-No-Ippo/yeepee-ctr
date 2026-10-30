"""Compare two submission CSVs: python compare.py NEW.csv REF.csv  -> Spearman/Pearson/max|diff| (rows matched on rowid)."""
import sys, numpy as np, polars as pl
from scipy.stats import spearmanr
a=pl.read_csv(sys.argv[1],schema_overrides={'rowid':pl.Utf8}); b=pl.read_csv(sys.argv[2],schema_overrides={'rowid':pl.Utf8})
j=a.join(b,on='rowid',suffix='_ref',how='inner'); assert j.height==a.height==b.height
x=j['predicted_probability'].to_numpy(); y=j['predicted_probability_ref'].to_numpy()
print(f'n={len(x)} spearman={spearmanr(x,y)[0]:.6f} pearson={np.corrcoef(x,y)[0,1]:.6f} max|dp|={np.abs(x-y).max():.5f} mean new={x.mean():.5f} ref={y.mean():.5f}')
