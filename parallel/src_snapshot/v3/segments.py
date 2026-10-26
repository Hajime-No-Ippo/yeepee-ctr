"""Validation AUC by user segment. Usage: python segments.py full25 25"""
import sys, os, json
import numpy as np, polars as pl
from train import load_table
from metric import global_auc
D = os.path.dirname(os.path.abspath(__file__))
name, cut = sys.argv[1], int(sys.argv[2])
va = load_table([f'val{cut}']); p = np.load(f'{D}/out/{name}_valpred.npy'); y = va['clicked'].to_numpy()
cold = va['us_cold'].to_numpy() == 1; noh = va['us_nohist'].to_numpy() == 1
segs = {'all': np.ones(len(y), bool), 'users_with_fit_rows': ~cold, 'history_only (no fit rows)': cold & ~noh,
        'empty_history': noh, 'cand_in_history': va['hist_in'].to_numpy() == 1, 'cand_not_in_history': va['hist_in'].to_numpy() == 0}
out = {}
for k, m in segs.items():
    out[k] = dict(rows=int(m.sum()), events=int(m.sum() // 10), ctr=float(y[m].mean()) if m.any() else None,
                  auc=global_auc(y[m], p[m]) if m.any() else None)
    print(f'{k:28s} rows={out[k]["rows"]:7d} ctr={out[k]["ctr"]:.4f} auc={out[k]["auc"]:.4f}')
json.dump(out, open(f'{D}/out/segments_{name}.json', 'w'), indent=1)
