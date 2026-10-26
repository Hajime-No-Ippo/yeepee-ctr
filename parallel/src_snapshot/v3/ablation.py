"""Ablation: add feature groups one at a time; binary LightGBM; val AUC at both cutoffs. Writes out/ablation.json."""
import json, os, sys, time
from train import load_table, fit, FOLDS
from features import feature_cols
D = os.path.dirname(os.path.abspath(__file__))
LR = float(os.environ.get('ABL_LR', '0.1')); TH = int(os.environ.get('ABL_THREADS', '4'))
steps = [['it'], ['it', 'hist'], ['it', 'hist', 'us'], ['it', 'hist', 'us', 'co'], ['it', 'hist', 'us', 'co', 'sv'],
         ['it', 'hist', 'us', 'co', 'sv', 'mf'], ['it', 'hist', 'us', 'co', 'sv', 'mf', 'ev'],
         ['it', 'hist', 'us', 'co', 'sv', 'mf', 'ev', 'tm']]
out = f'{D}/out/ablation.json'
res = json.load(open(out)) if os.path.exists(out) else {}
for cut in (25, 26):
    tr = load_table(FOLDS[cut]); va = load_table([f'val{cut}'])
    for g in steps:
        key = f'{cut}:' + '+'.join(g)
        if key in res: continue
        t0 = time.time()
        feats = feature_cols(tr, g)
        m, best, auc = fit(tr, va, feats, 'binary', LR, TH)
        res[key] = dict(auc=auc, best_iter=best, n_feats=len(feats), secs=round(time.time() - t0))
        print(key, res[key], flush=True)
        json.dump(res, open(out, 'w'), indent=1)
