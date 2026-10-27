# v12 NOTES (v11 + DW feature families)
## 21:10 step 1: features
- mkdw.py: reusable.all_recommended (42 cols: impression_taste+slate_centrality+skip_contrast) for rows of f1..f6, val23/25/26, eval -> feat_dw/*.parquet. Fit cutoffs: f_k=J20+k*DAY, val23/25/26, eval=J27 (train ts<X only). Eval: eval events appended to a Data copy only for own-history lookups; snapshot is exactly train events (assert). 1453 eval users unseen in train -> empty user row (NaN features).
- val25 features reproduce DW cache exactly (42 cols, 0 mismatches). ~60-75s per cut.

## step 2: method (b) full retrain (LightGBM dwbig+dw_s, MLP dwmlp x3 seeds) -- done
b_full = dwbig:1 + dw_s:1 + 2*dwmlp:3 (seeds 0,1,2), same recipe as v11
 v11: C25 .81524  C23 .79424  | b_full: C25 .81723 (+.00199)  C23 .79566 (+.00142)
 LGB-only swap (v11 MLP kept): C25 .81668, C23 .79523
 timings: LGB dw_s ~1min, dwbig ~4min per cut; MLP ~3-4min per run (see timings.jsonl)

## step 3: method (a) residual boost (resid.py oof, 5-fold grouped by user, seeds 0,1) -- done
 C25 +.00152/+.00156 vs v11 ; C23 +.00131/+.00129 ; C26 +.00156/+.00156   (matches DW's +.00146/+.00143)
 => (b) full retrain wins on all three cuts: C25 +.00199, C23 +.00142, C26 +.00170 (C26 on MLP seed 0 only: .82604 vs .82434).
 Chosen: (b). Second-seed confirmation: LGB seed 1 for dw_s/dwbig and v11 sC_s/sCbig on cuts 25/23 (b3.sh); MLP per-seed comparison.

## step 4: SUBMISSION BUILT (method b, seed-0 recipe) 21:45 Dublin
 spec: dwbig:1 + dw_s:1 + 2*dwmlp:3 (seeds 0,1,2) at cut 27; raw logit mean -> sigmoid
 out/submission_v12.csv: 529,390 rows, rowids == v8 set, unique, p in [0.00026,0.831], no NaN. Spearman vs v11 .9938, vs v8 .9809, vs v10 .9773
 gate: C25 .81723 / rw .81722 < .83 -> refused (exit 1) as expected
 candidate zip: out/submission_v12_candidate.csv.zip sha256 1dcf21846cf7fa8c7de356f630d7888b6ff4f739299c544354df57f6d6d16506
 csv sha256 37898b6f51715f4aa19400366545aee5da715332fce5ccfe3b3f421229c4b10b
 NEXT: second-seed confirmation (b3.sh, LGB seed 1 dw vs v11 baselines, cuts 25/23). b3.sh was never launched (shell aborted).

## step 5: second-seed confirmation (LGB seed 1; cuts 25/23) -- PASS 21:~ Dublin
 v11 (LGB s1, MLP s1)        C25 .81523 C23 .79424 | v12b (LGB s1, MLP s1) C25 .81721 C23 .79557  (+.00198 / +.00133)
 v11 (LGB s1, MLP s0-2)      .81529 / .79423       | v12b                  .81727 / .79562        (+.00198 / +.00139)
 v11 (LGB s0+1, MLP s0-2)    .81529 / .79429       | v12b                  .81728 / .79568        (+.00199 / +.00139)
 accept rule (C25 up, C23 not down >.0002) holds for both seeds. Submission uses seed-0 LGB + 3 MLP seeds (not the 2-seed LGB average).
 runtimes (sum of wall): DW feature builds ~534s, dw_s 361s, dwbig 1425s, dwmlp 599s(+prep), seed-1 baselines 415s, resid oof ~2min. Whole v12 ~1h45 incl. waits.
 scratch cleaned (__pycache__); kept feat_dw, ens, logs.
