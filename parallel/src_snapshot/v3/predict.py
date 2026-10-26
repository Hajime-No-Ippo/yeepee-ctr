"""Final fit + prediction.
  1. (re)build features for folds f1..f6 (cutoffs Jan21..Jan26) and eval (cutoff Jan27) if missing (--rebuild forces).
  2. train binary LightGBM on f1..f6 for n_iter = best_iter(cutoff 26 val) * scale (more data -> a few more rounds).
  3. predict eval, write submission (clip [0,1], NaN->0.0629, %.8g), order = example_submission.csv.
Config read from out/best_config.json (written after validation); defaults below if missing.
Usage: python predict.py [--rebuild] [--out /workspace/huawei/submission_v2.csv]
"""
import argparse, json, os, subprocess, sys, time
import numpy as np, polars as pl
from train import load_table, fit, FOLDS, importance
from features import feature_cols, GROUPS
D = os.path.dirname(os.path.abspath(__file__))

def write_submission(rowid, pred, path, example='/workspace/huawei/example_submission.csv'):
    p = np.nan_to_num(np.asarray(pred, dtype=np.float64), nan=0.0629, posinf=1.0, neginf=0.0).clip(0, 1)
    df = pl.DataFrame({'rowid': rowid, 'predicted_probability': p})
    ex = pl.read_csv(example, schema_overrides={'rowid': pl.Utf8}).select('rowid')
    df = ex.join(df, on='rowid', how='left', maintain_order='left').with_columns(pl.col('predicted_probability').fill_null(0.0629))
    with open(path, 'w', encoding='utf-8', newline='') as f:
        f.write('rowid,predicted_probability\n')
        f.writelines(f'{r},{v:.8g}\n' for r, v in zip(df['rowid'].to_list(), df['predicted_probability'].to_list()))
    return df

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--rebuild', action='store_true')
    ap.add_argument('--out', default='/workspace/huawei/submission_v2.csv')
    ap.add_argument('--threads', type=int, default=6)
    a = ap.parse_args()
    t0 = time.time()
    cmd = [sys.executable, f'{D}/build_feats.py', 'all'] + (['--force'] if a.rebuild else [])
    subprocess.run(cmd, check=True)
    t_feat = time.time() - t0
    cfg_path = f'{D}/out/best_config.json'
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    groups = cfg.get('groups', GROUPS); drop = cfg.get('drop', [])
    lr = cfg.get('lr', 0.05); n_iter = int(round(cfg.get('best_iter', 400) * cfg.get('scale', 1.15)))
    params = cfg.get('params', None); folds = cfg.get('folds', FOLDS[27])
    tr = load_table(folds)
    feats = [f for f in feature_cols(tr, groups) if not any(f.startswith(d) for d in drop)]
    print(f'final fit: {len(tr)} rows, {len(feats)} feats, lr={lr}, n_iter={n_iter}', flush=True)
    t1 = time.time()
    m, _, _ = fit(tr, None, feats, 'binary', lr, a.threads, max_rounds=n_iter, params=params)
    t_fit = time.time() - t1
    m.save_model(f'{D}/out/final_model.txt')
    ev = pl.read_parquet(f'{D}/feat/eval.parquet')
    pred = m.predict(ev.select(feats).to_numpy().astype(np.float32))
    write_submission(ev['rowid'], pred, a.out)
    info = dict(feature_build_secs=round(t_feat), fit_secs=round(t_fit), total_secs=round(time.time() - t0), n_iter=n_iter,
                n_train_rows=len(tr), n_feats=len(feats), top15=importance(m))
    json.dump(info, open(f'{D}/out/final_info.json', 'w'), indent=1)
    print(json.dumps(info), flush=True)
    subprocess.run([sys.executable, f'{D}/validate_submission.py', a.out], check=True)
