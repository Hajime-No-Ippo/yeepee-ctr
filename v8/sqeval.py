"""python sqeval.py DIR CUT [CTL=pp] -> standalone AUC of sq_logit and blend with ens/{CTL}_blend_c{CUT}.npy on the rows present in DIR/val{CUT}."""
import sys, json, numpy as np, polars as pl
from sklearn.metrics import roc_auc_score
d, c = sys.argv[1], int(sys.argv[2]); ctl = sys.argv[3] if len(sys.argv) > 3 else 'pp'
v = pl.read_parquet(f'feat_v3/val{c}.parquet', columns=['rowid', 'clicked']).with_columns(pl.Series('z', np.load(f'ens/{ctl}_blend_c{c}.npy')), pl.int_range(pl.len()).alias('i'))
j = v.join(pl.read_parquet(f'{d}/val{c}.parquet'), on='rowid', how='inner'); y = j['clicked'].to_numpy(); z = j['z'].to_numpy(); s = j['sq_logit'].to_numpy()
r = dict(dir=d, cut=c, ctl=ctl, n=len(y), sq=round(roc_auc_score(y, s), 5), ctl_auc=round(roc_auc_score(y, z), 5), corr=round(float(np.corrcoef(s, z)[0, 1]), 3))
sz = (s - s.mean()) / s.std() * z.std() + z.mean()
for w in (0.05, 0.1, 0.2, 0.3): r[f'w{w}'] = round(roc_auc_score(y, (1 - w) * z + w * sz), 5)
print(json.dumps(r)); open('sq_results.jsonl', 'a').write(json.dumps(r) + '\n')
r2 = dict(dir=d, cut=c, ctl=ctl, kind='rawlogit')
for w in (0.1, 0.2, 0.3, 0.4): r2[f'w{w}'] = round(roc_auc_score(y, (1 - w) * z + w * s), 5)
print(json.dumps(r2)); open('sq_results.jsonl', 'a').write(json.dumps(r2) + '\n')
