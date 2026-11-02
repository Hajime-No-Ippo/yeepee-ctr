# v21cat STATUS (started 2026-11-02 01:52 Dublin)
Goal: ID-cross GBDT with ordered/time-respecting target encoding, diverse from M/A. CatBoost not installed -> LightGBM + hand-built TE (mkcat.py). Cores 4-7, 2 threads.
## Design
- Data has NO item attributes: only user, item, time, history. So "categories" = latent item clusters (MiniBatchKMeans K=48/256 on click-SVD item vectors at each cutoff); user cluster = cluster of the centroid of own history.
- TE: for cutoff X (day k), counts/CTR from train events ts<X only (the fold design makes this identical to ordered TE: every target row is after X). Eval uses all train. Crosses: item, user, item×{tb,3h-bin,daytype}, user×item, user×{tb,dt}, user×icl{48,256}, ucl×item, ucl×icl, icl×hour, last-hist×cand, any-of-last5-hist×cand (mean/max/min/lift), lasthist-cluster×cand, + in-history counts, hlen, staleness. 59 cols -> cache/{table}.npy, names.json.
- Train tables = f{cut-24..cut-21}, validate val{cut} (same as v21seq). Scripts: mkcat.py, train.py (variants ids | num | numte), ev.py.
## Findings log
## RESULTS (single seed, LightGBM; lr .03, 63 leaves; C25@900 rounds, C23/C26@600)
| variant | C23 | C25 | C26 | rho_M (C23/25/26) |
|---|---|---|---|---|
| ids (TE crosses + hour/dt, no numeric) | - | .5475 (best .5491@300) | - | C25 .04 |
| num (top-80 numeric, control) | - | .8165 | - | C25 .981 |
| numte (numeric + TE crosses) | .7965 | .8164 | .8249 | .976/.980/.982 (rho_A same) |
Gain (x1e-5 AUC) numte, w = weight of cat in (1-w)*base+w*cat:
- on .4z+.6M: C23 w.1:+7 w.15:+7 w.2:+6 | C25 w.05:+3 w.1:+4 w.15:+3 w.2:0
- on stack .3z+.3M+.4D: C23 w.1:+10 w.15:+12 w.2:+13 | C25 w.05:+2 w.1:+3 w.15:+1 w.2:-2
- stated weight w=0.10: gain25=+4e-5 (+3e-5 on DCN stack), gain23=+7e-5 (+10e-5 on DCN stack)
## VERDICT: FAILS the bar (needs >=+3e-4 on both cuts, rho<=.97). TE crosses add nothing: num vs numte identical (.8165 vs .8164); ids-only is ~.55 (no within-slate signal; items have no attributes). Not built for C27 (no cat_eval.npy).
## Fold predictions (raw LGB scores, float32, val-table row order = v21seq ens order)
- out/cat_numte_c23.npy (501550) , out/cat_numte_c25.npy (439640), out/cat_numte_c26.npy (351380)
Logs: logs/ev_numte.txt, logs/train_log.jsonl. Cache: cache/*.npy
