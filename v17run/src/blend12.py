"""v12 final blend: z = (dwbig@1000 + dw_s@500 + 2 * mean_over_3_seeds(dwmlp@epoch4)) / 4 (logit blend 1:1:2), p = sigmoid(z).
python blend12.py CUT : CUT=27 -> eval_pred.parquet (rowid, p); CUT=25 -> prints/saves the local AUC on val25."""
import sys, json, numpy as np, polars as pl
from cl import matrix, BASE
cut = int(sys.argv[1]); vn = 'eval' if cut == 27 else f'val{cut}'
_, m, _ = matrix(vn, 1.0, extra=BASE)
big = np.load(f'ens/dwbig_c{cut}_s0.npy')[:, 0].astype(np.float64); sm = np.load(f'ens/dw_s_c{cut}_s0.npy')[:, 0].astype(np.float64)
ml = np.mean([np.load(f'ens/dwmlp_c{cut}_s{s}.npy')[:, 3].astype(np.float64) for s in (0, 1, 2)], 0)
assert len(big) == len(sm) == len(ml) == m.height
z = (big + sm + 2 * ml) / 4; p = 1 / (1 + np.exp(-z)); assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
print('corr big/small/mlp', round(float(np.corrcoef(big, sm)[0, 1]), 4), round(float(np.corrcoef(big, ml)[0, 1]), 4))
if cut == 27:
    pl.DataFrame({'rowid': m['rowid'], 'p': p}).write_parquet('eval_pred.parquet'); print('wrote eval_pred.parquet', m.height)
else:
    from sklearn.metrics import roc_auc_score
    y = m['clicked'].to_numpy(); r = {k: round(roc_auc_score(y, v), 5) for k, v in dict(dwbig=big, dw_s=sm, mlp=ml, blend=z).items()}
    print('LOCAL AUC cut', cut, r); json.dump(r, open(f'local_auc_c{cut}.json', 'w'))
