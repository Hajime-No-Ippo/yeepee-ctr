# v12x NOTES (XGBoost / lambdarank / max_bin experiments on v12 features)
- Setup: xgboost 3.4.1 in v12x/pylib (uv --target; main venv untouched). xens.py = XGB on v12 matrix (435 cols after dropping sq_(pdot|pcos|mdot|mcos); the "562" quoted by parent = 435 used by dwbig; same cl.matrix, same folds). cl.py = copy of v12/cl.py that globs the base cache by table name (original key includes file mtimes, changed by the snapshot restore). +-inf clipped to +-1e15 (xgboost rejects inf).
- XGB params: hist, lossguide max_leaves127, eta .03, subsample .8, colsample_bytree .5, min_child_weight 30, lambda 30, max_bin 255.
- [04:12] Round pick on C25 train folds f1-f3, early stop on hold fold f4 (not val): best_iter 403 (hold AUC .80985), 553 s. Declared: use 450 rounds for all cuts (saved 300/450/600). 
- [04:12] c25 clean solution run: blend .81727 (see v12/sol_work/verify_stats_c25.json).
- [04:15] queued: run_xgb.sh (xgb c25/c23 seeds 0,1 @450 rounds), then run_q2.sh (lgb255 sanity, lambdarank, rank_xendcg, max_bin 1023/511; c25+c23). logs in v12x/logs, results ens/xlog.jsonl, timings ens/timings.jsonl
- [04:15] XGB c25 s0: @300 .81605 @450 .81619 @600 .81622 (dwbig .81658); fit 368s. Parent queued: combined v12+propensity+tm_prod retrain (+ablations, + 'in-history x taste' interaction) after q2.
- [04:35] built per-fold features with v12x/mk/mkfeat.py (f1-f4,val23,val25): feat_prop (28; matches clean25 file except ~110 tie rows), feat_tm (tm_prod0-11+tail; matches DW cache val25 1e-4), feat_tmx (tm_inh,tm_cos,_in,_out,_dm,tm_inh_n interaction). cl.py now prefers v12x/<dir> for extra feature dirs.
- [04:40] queued run_q3.sh: dwbig + {prop+tm, tm, prop, tm+tmx, all} on C25,C23 (tags f_*).

## 04:32 XGB result (hist CPU, lossguide 127 leaves, eta .03, ss .8, cs .5, mcw 30, lambda 30, 600 rounds; primary = @450 from early-stop pick best_iter 403 on f4 hold)
C25 alone: s0 .81619 / s1 .81634 (@450) vs dwbig .81658 / .81652 (gap -.0004 / -.0002; within noise band)
C23 alone: s0 .79424 / s1 .79399 (@450) vs dwbig .7946 / .79458 (gap -.0004 / -.0006)
Fit times: C25 ~6.7 min, C23 ~3.6 min (8 cores).
4th member, weights 1:1:2:w (@450): C25 w=.5/1/2: s0 .81726/.81725/.81721, s1 .81728/.81729/.81726 (v12 .81723)
  C23: s0 .79572/.79573/.79568, s1 .79568/.79567/.79558 (v12 .79566)
Max delta w=1: C25 +.00002/+.00006, C23 +.00007/+.00001 => noise; FAILS accept rule. No v12x submission built.
## 04:34 queue q2/q3 killed (lrank C25 just started, lgb255 done) for priority p33 fingerprint + XGB probe job; restart later from lrank
## queued at END: CatBoost experiment /workspace/huawei/v15cat (parent msg 04:37)
