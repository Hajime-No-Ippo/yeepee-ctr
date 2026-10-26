"""python blend.py [--seq] : per-component and blended AUC on val25/val26. Components: seed-averaged raw scores at chosen round column."""
import sys, glob, json, numpy as np, polars as pl
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from cl import matrix
COL = {'base': 1, 'wide': 1, 'slow': 2, 'nov2': 1, 'estrc': 1}   # 400 rounds (slow: 500)
def comps(cut, var_meta):
    out = {}
    for t, ci in COL.items():
        fs = sorted(glob.glob(f'ens/{t}_c{cut}_s*.npy'))
        if fs: out[t] = np.mean([np.load(f)[:, min(ci, np.load(f).shape[1]-1)] for f in fs], 0); out[t+'_n'] = len(fs)
    return out
res = {}; D = {}
for cut in (25, 26):
    _, m, _ = matrix(f'val{cut}', 1.0); y = m['clicked'].to_numpy()
    _, m2, _ = matrix(f'val{cut}', 1.0, variant='stale', extra=('feat_rc',))
    assert (m['rowid'] == m2['rowid']).all()
    c = comps(cut, m); nn_ = {k: c.pop(k) for k in list(c) if k.endswith('_n')}
    if '--seq' in sys.argv:
        s = m.select('rowid').join(pl.read_parquet(f'ens/seq_c{cut}.parquet'), on='rowid', how='left')['seq_logit'].to_numpy(); c['seq'] = s
    r = {k: round(roc_auc_score(y, v), 5) for k, v in c.items()}; r['nseeds'] = nn_
    r['REF_v3_base_s012_r400'] = round(roc_auc_score(y, np.mean([np.load(f'ens/base_c{cut}_s{i}.npy')[:, 1] for i in range(3)], 0)), 5)
    sub = [k for k in c if k != 'nov2']
    r['mean_logit_no_nov2'] = round(roc_auc_score(y, np.mean([c[k] for k in sub], 0)), 5)
    if 'base' in c:
        b1 = c['base'][:, None] if False else None
    ks = list(c); Z = np.stack([c[k] for k in ks], 1)
    r['mean_logit_all'] = round(roc_auc_score(y, Z.mean(1)), 5)
    r['rank_avg_all'] = round(roc_auc_score(y, np.mean([rankdata(Z[:, i]) for i in range(Z.shape[1])], 0)), 5)
    D[cut] = (y, Z, ks); res[cut] = r
# logistic blend: fit on one cutoff, apply on the other
for a_, b_ in ((25, 26), (26, 25)):
    ya, Za, ks = D[a_]; yb, Zb, kb = D[b_]
    if ks != kb: continue
    lr = LogisticRegression(C=1.0, max_iter=500).fit(Za, ya)
    res[b_][f'logit_blend_fit{a_}'] = round(roc_auc_score(yb, Zb @ lr.coef_[0]), 5); res[b_][f'w_fit{a_}'] = dict(zip(ks, np.round(lr.coef_[0], 3).tolist()))
print(json.dumps(res, indent=1)); json.dump(res, open('ens/blend_results.json', 'w'), indent=1)
