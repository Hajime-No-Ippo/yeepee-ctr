import sys, json, glob, numpy as np, polars as pl
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from cl import matrix
def evaucs(df,s):
    d=df.with_columns(s=pl.Series(s))
    E=d.group_by('eid').agg(pl.col('clicked').max(),pl.col('s').max())
    W=d.group_by('eid').agg((pl.col('clicked').max()>pl.col('clicked').min()).alias('mx'),pl.col('clicked'),pl.col('s')).filter('mx')
    R=d.with_columns(r=pl.col('s').rank('average').over('eid')).group_by('eid').agg(np_=pl.col('clicked').sum(),n=pl.len(),rs=(pl.col('r')*pl.col('clicked')).sum())
    R=R.filter((pl.col('np_')>0)&(pl.col('np_')<pl.col('n')))
    wa=((R['rs']-R['np_']*(R['np_']+1)/2)/(R['np_']*(R['n']-R['np_']))).mean()
    return dict(glob=round(roc_auc_score(d['clicked'],d['s']),5),event=round(roc_auc_score(E['clicked'],E['s']),5),within=round(wa,5))
COL = {'base': 1, 'wide': 1, 'slow': 2, 'estrc': 1}
out = {}; G = {}
for cut in (25, 26):
    _, m, _ = matrix(f'val{cut}', 1.0); y = m['clicked'].to_numpy()
    s = m.select('rowid').join(pl.read_parquet(f'ens/seq_c{cut}.parquet'), on='rowid', how='left')['seq_logit'].to_numpy()
    assert np.isfinite(s).all()
    c = {t: np.mean([np.load(f)[:, min(ci, np.load(f).shape[1]-1)] for f in sorted(glob.glob(f'ens/{t}_c{cut}_s*.npy'))], 0) for t, ci in COL.items()}
    g = np.mean(list(c.values()), 0); base = c['base']
    df = m.select('eid', 'clicked')
    r = dict(seq=evaucs(df, s), gbdt_blend=evaucs(df, g),
             pearson_seq_base=round(float(np.corrcoef(s, base)[0, 1]), 4), spearman_seq_base=round(float(spearmanr(s, base)[0]), 4),
             pearson_seq_blend=round(float(np.corrcoef(s, g)[0, 1]), 4))
    sz = (s - s.mean()) / s.std() * g.std()
    r['w_sweep'] = {w: round(roc_auc_score(y, g + w * sz), 5) for w in (0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)}
    G[cut] = r['w_sweep']; out[cut] = r
for a, b in ((25, 26), (26, 25)):
    w = max(G[a], key=G[a].get); out[b][f'crossfit_w_from{a}'] = [w, G[b][w]]
print(json.dumps(out, indent=1)); json.dump(out, open('ens/seq_analysis.json', 'w'), indent=1)
