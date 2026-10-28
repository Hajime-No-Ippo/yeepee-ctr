# dw_levers findings (Europe/Dublin, 2026-10-28)
Baseline = v12 z12 blend (C25 0.81723, C23 0.79566). All numbers on val rows (user's first event after cutoff).

## AUC decomposition (decomp.json)
| | C25 | C23 |
|---|---|---|
| Global AUC | 0.8172 | 0.7957 |
| Within-event pooled AUC | 0.8258 | 0.8043 |
| Event "any click" AUC of 1-prod(1-p) | 0.673 | 0.650 |
| Oracle event label + model within-event | 0.891 | 0.879 |
| Within-event pair share of all pairs | 2e-5 | 2e-5 |
Global AUC is almost entirely between-event, so it is set by cross-event calibration (a mix of item quality + event-level click propensity). Headroom sim (headroom_sim.json): event AUC 0.673->0.709 = +0.003, ->0.758 = +0.0075, ->0.80 ≈ +0.012. A +0.012 gain from the event lever alone needs event AUC ≈0.80 (from 0.67), not realistic: v13 exp1 event residual models were negative, my choice-model "no-click" head has event AUC 0.62 (<0.673).

## Ranked levers (ΔAUC over z12, C25)
| # | lever | result |
|---|---|---|
| 1 | Event-level model (headroom + v13 + choice outside option) | long shot, needs event AUC +0.13; every attempt ≤0 |
| 2 | Choice model w/ outside option, PyTorch end-to-end (choice.py) | standalone 0.778 (lr1e-3, 1 epoch; best-by-holdout), blend with z: -0.0001 (w=.1) to -0.003 (w=.4); LGBM residual stacking with lp/slate-rank/outside logit vs control with only z-derived feats: +0.00008 [95% CI +0.00002,+0.00015] (event bootstrap) = negligible |
| 3 | Two-tower/seq/LightGCN | Not built separately: choice model is effectively this (SVD-init item/user-hist embeddings + softmax) and reached only 0.78; frozen SVD embeddings 0.753; attention 0.768; hist dropout / l2 / lr sweeps (sweep_choice.py, choice_sweep.jsonl) all 0.75-0.778 and overfit after 1 epoch (holdout loss min at epoch 0). Earlier DIN 0.534. Not worth blending |
| 4 | Rank/linear blend diversity (blend_div.py) | all 44 existing ens members (dw_s, dwbig, dwmlp, mlp2, sC, sCbig × seeds) correlate .97-.995 with z12; best incremental blend +0.00007 (dwbig seed1). Ensemble is saturated |
| 5 | Error analysis / cross-segment recalibration (seg.py, recal.py) | Per-segment Platt recalibration by us_hlen, it_imp, us_nev (+ crosses), user-grouped CV: Δ -0.0003..+0.00006 → score already calibrated across segments, no headroom |

Stale rows: `tm_stale` in val parquet is not hours (all >48), so C23 stale slice was not separable via it; no model in the ensemble or the choice model is better on any slice—ens members all correlated ≥.97.

## Where the weakness is (AUC within segment)
C25: hist len ≤6: 0.755 (18% rows) vs hist 17-20: 0.836 (49%); users with ≤2 events 0.767 vs ≥10 events 0.84; low-impression items (it_imp<61) 0.78 vs >237 0.84. C23: hlen≤5 0.730 (21%), us_nev≤1 0.751 (24%), it_imp≤36 0.757. So cold users/cold items are the weak spots — information-limited (no history), not model-limited; every learned model fails there too.

## Honest notes / failures
- First choice-model run was invalid (val AUC 0.496): scores were in D.RI order (rows sorted by item), val labels in parquet order. Fixed with align()/to_val() in choice.py.
- Choice model overfits: 50k×32 free item embeddings on ~490k events; best-by-holdout is epoch 0-1. l2 / hist-dropout / low emb-lr don't get above 0.778 (SVD frozen 0.753).
- Everything ≥ plateau; no lever found with ≥+0.001. Gap to 0.83 is not closable by blend/stack/neural diversity on the current information set.

## Files
common.py, dwcommon.py, decomp.py/json, headroom_sim.py/json, choice.py (CM, fit_choice, score, align, to_val), run_choice.py, sweep_choice.py + cfg1/2.txt + choice_sweep.jsonl, stack_choice.py (LGBM residual stacking test), blend_div.py, seg.py, recal.py, logs/.
Run e.g.: /workspace/venv/bin/python stack_choice.py 25 '{"d":32,"ep":1,"lr":0.001}' ep1
