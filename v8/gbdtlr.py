"""S3 GBDT+LR member: one-hot leaf indices of the base model (first NT trees) -> L2 logistic regression on the same lite training rows.
python gbdtlr.py CUT TAG [--nt 150] [--C 0.05] [--rows 400000]. Saves ens/{TAG}_lr_c{CUT}.npy (raw logit)."""
import sys, json, gc, time, argparse, numpy as np, lightgbm as lgb, scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from cl import matrix
ap = argparse.ArgumentParser(); ap.add_argument('cut', type=int); ap.add_argument('tag'); ap.add_argument('--lr', default='feat_lr2,feat_tt')
ap.add_argument('--nt', type=int, default=150); ap.add_argument('--C', type=float, default=0.05); ap.add_argument('--rows', type=int, default=400000); ap.add_argument('--nfolds', type=int, default=4)
a = ap.parse_args(); t0 = time.time(); extra = tuple(x for x in a.lr.split(',') if x)
mdl = lgb.Booster(model_file=f'ens/{a.tag}_b_c{a.cut}_s0.txt'); fl = mdl.feature_name(); NL = 63
trn = [f'f{k}' for k in range(max(1, a.cut - 20 - a.nfolds), a.cut - 20)]
Xs, ys = [], []
for n in trn:
    X, m, f = matrix(n, 1.0, extra=extra); X = X[:, [f.index(c) for c in fl]]; Xs.append(X); ys.append(m['clicked'].to_numpy()); del X, m; gc.collect()
X = np.concatenate(Xs); y = np.concatenate(ys); del Xs
rs = np.random.RandomState(7); sel = rs.rand(len(y)) < min(1.0, a.rows / len(y)); X = X[sel]; y = y[sel]
def onehot(X):
    L = mdl.predict(X, pred_leaf=True, num_iteration=a.nt).astype(np.int64) + np.arange(a.nt)[None, :] * NL
    n = L.shape[0]; return sp.csr_matrix((np.ones(L.size, np.float32), L.ravel(), np.arange(0, L.size + 1, a.nt)), shape=(n, a.nt * NL))
H = onehot(X); del X; gc.collect(); t1 = time.time()
lr = LogisticRegression(C=a.C, solver='liblinear', max_iter=200).fit(H, y); t_fit = time.time() - t1
vn = 'eval' if a.cut == 27 else f'val{a.cut}'
Xv, mv, fv = matrix(vn, 1.0, extra=extra); Xv = Xv[:, [fv.index(c) for c in fl]]
s = lr.decision_function(onehot(Xv)).astype(np.float32); np.save(f'ens/{a.tag}_lr_c{a.cut}.npy', s)
r = dict(cut=a.cut, tag=a.tag, nt=a.nt, C=a.C, n=int(len(y)), fit_s=round(t_fit), total_s=round(time.time() - t0))
if a.cut != 27:
    yv = mv['clicked'].to_numpy(); r['auc_lr'] = round(roc_auc_score(yv, s), 5)
    try:
        z = np.load(f'ens/{a.tag}_blend_c{a.cut}.npy'); r['blend'] = round(roc_auc_score(yv, z), 5)
        sz = (s - s.mean()) / s.std() * z.std() + z.mean()
        for w in (0.1, 0.2, 0.3): r[f'w{w}'] = round(roc_auc_score(yv, (1 - w) * z + w * sz), 5)
    except FileNotFoundError: pass
print(json.dumps(r), flush=True); open('scr_results.jsonl', 'a').write(json.dumps(r) + '\n')
