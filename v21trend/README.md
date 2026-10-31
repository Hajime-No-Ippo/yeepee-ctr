# v21trend -- trend / cold-start / two-hop / cluster / decayed-TE feature screens (variant A, train-only, leak-free)
Nothing outside this folder was modified; no leaderboard upload; no probe built (no feature set cleared the gate).

Proxy: v21roll/rollval.py recipe (LGB 300 rounds, lr .06, 63 leaves, neg-sub .35, 4 threads), BASE5 + feat_Ap (460 feats) + new set.
Folds: C23 = M3 on val23, C25 = M5 on val25, C26 = M6 on f6; K4 = plain f3,f4,f5,f6. Noise: paired Poisson bootstrap by event (sd ~.0002 per fold, ~.00012 on mean3); seed-1 baseline (coA_s1) gives seed noise (mean3 -.00003, per fold up to .0003).

## Feature scripts
- mktrend.py -> feat_trI (19), feat_trU (25): smoothed item trend (k=20,100 recent 1d/2d vs long fit-window rate, EB, share, slate rank/z) and user-level trend (+ own-history trends).
- mkphase2.py -> feat_cs (11 cold-start flags), feat_co2 (14, two-hop cosine/PPMI-walk scores, max10/sum10/last/mean3 + slate rank/z), feat_cl (30, cluster k=64/256 on train-only SVD32 click embeddings, lag-1 transition lift from last/mean3/mean10/max10 history, ctr, marg, slate rank/z).
- mkdecay.py -> feat_dec (51): decayed CTR, hl 6/12/24h, item (k=20,100, imp volume, diff vs long ctr), user (k=50,200, activity, clicks), user x item (ctr, imp, clk), slate rank/z/gap on 6 main ones. Shares data/target-frame code with mktrend.py.
- rollval_t.py (fit/score), score3.py (C23/C25/C26 + segments + bootstrap), run_*.sh, results in score3_p1.json / score3_p2.json, logs/score_p*.log.

## Results (AUC; delta vs coA, +-bootstrap sd)
| set | C23 | C25 | C26 | mean3 | K4 |
|---|---|---|---|---|---|
| coA | .79483 | .81628 | .82532 | .81214 | .81497 |
| coA_s1 (seed) | +.00015 | +.00006 | -.00031 | -.00003 | -.00011 |
| trI | +.00011 | -.00019 | -.00001 | -.00003+-.00011 | -.00013 |
| trU | -.00015 | -.00018 | -.00045 | -.00026+-.00011 | -.00028 |
| trIU | +.00014 | -.00033 | -.00012 | -.00010+-.00011 | -.00014 |
| cs (2a flags) | -.00014 | -.00018 | -.00028 | -.00020 | -.00026 |
| co2 (2b two-hop) | +.00016 | -.00018 | -.00022 | -.00008 | -.00008 |
| cl (2b clusters) | +.00019 | -.00018 | -.00034 | -.00011 | -.00022 |
| dec (2c) | -.00027 | -.00047 | -.00023 | -.00033+-.00012 | -.00033 |
Segments/staleness in logs/score_p2.log: all deltas within noise. Largest: stale>=24h rows on C23 (cl +.0005+-.0005, cs +.0006+-.0005, dec -.0005), C25 stale (co2 +.0006+-.0004); cold users +.0008..+.002 on C23/C25 for several sets but sd ~.001-.0015 and negative on C26.
Cold-start flags: cold items essentially do not occur (cs_it_noimp24/lowimp20 share ~0 on every fold; noimp6 0.5%); cold users 1.7-10%; never-shown pairs 97-98.5% (so never-shown flag ~ constant). The original cold columns are zero-filled, not NaN, so the flags were largely redundant.
Gate (>= +.0002 mean on folds, positive on C23 and C25): not met by any set. No v21list A/B retrain, no probes, no blend probe.

## Runtime impact (if one were adopted)
feat_cs/co2/cl: ~10-30 s per cut incl. eval (25 s), single core; feat_dec ~3 s per table; trI/trU a few s per table. Negligible vs budget; training cost +3-10% (more features).
MLP sequence branch (not done): needs an end-to-end history encoder in mlp17, many times slower per epoch on 4 CPU cores; likely breaks the 15-min budget unless history is precomputed/frozen.
