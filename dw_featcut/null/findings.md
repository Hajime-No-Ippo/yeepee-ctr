# Null-importance test on v2_/ev_ columns (v19 set, S already dropped)

Date 2026-10-30 (Europe/Dublin = UTC). Everything in dw_featcut/null/, nothing else written.

## Method
- Base: v19 dwbig recipe (lr .015, 127 leaves, mdl 200, ff .5, bf .8/1, l2 30, 1000 rounds), variant A, DROP sq_* and S (83 cols, v19cut/drop_S_A.txt) applied -> 401 cols, of which 148 are v2_ (76) / ev_ (72).
- C25 fit window (f1-f4, 1,402,200 rows, full, no subsample), val25 439,640 rows.
- Real fit (seed 42, AUC .81737 = v19cut A_S s0 exactly) + 5 null fits: label blocks (10 rows/event, verified event-major) permuted across events, so per-event label structure is kept and feature<->label link broken. Null AUC .494. Gain per feature recorded (imp_real.json, imp_null0-4.json, null_table.json).
- Caveat: null total gain is ~30% of real (1.30M vs 4.34M) because 127 leaves x 1000 rounds overfits noise; the null gain is therefore a large, conservative bar. So thresholds flag a lot of columns (73% at p90).
- Interpretation: "real gain < null p90" = gain not distinguishable from what a pure-noise column of that kind soaks up by overfitting; says nothing about redundancy-with-other-columns, so validated empirically.
- Python: /workspace/venv/bin/python is gone from the box; used /workspace/venv_v17/bin/python (lightgbm 4.7.0), same results as before.

## Flagged counts (148 v2_/ev_ cols)
| threshold | cols | v2_/ev_ | real gain share of total |
|---|---|---|---|
| real < null p90 (a_p90) | 108 | 48/60 | 6.57% |
| real < null mean (b_mean) | 95 | 42/53 | 5.77% |
| real/null_mean < 0.8 (c_r08, extra, very noise-like) | 29 | 10/19 | - |
| ratio<2 / <3 / <5 / <10 | 127 / 136 / 141 / 146 | | 8.0/9.2/9.6/10.4% |
| Actual-vs-Null score<0 | 102 | 47/55 | 6.26% |
Lowest ratios (real/null_mean): ev_hist_noclk_max .2, ev_sv_k64_ucos_rk .5, v2_svd48_cos_rk .6, ev_mf_score_rk .6, ev_co_tr_max5_z .7, ev_sv_k64_ccos_rk .7, ev_co_cos_max20_mean .7, ev_it_ctr24_z .7, v2_svd48_cos_dm .7.
Sets are nested: c_r08 < b_mean < a_p90. Lists in sets.json.

## Validation vs S baseline (full folds, 2 seeds, 4 threads, AUC delta x1e-5, seed 0 / seed 1 / mean)
| set (cols dropped beyond S) | C25 s0 | C25 s1 | C25 mean | C23 s0 | C23 s1 | C23 mean | keep rule (>= -3 both) |
|---|---|---|---|---|---|---|---|
| a_p90 (108) | +23 | -11 | +5.9 | +9 | +36 | +22.5 | PASS |
| b_mean (95) | +6 | -1 | +2.8 | -19 | +32 | +6.4 | PASS |
| c_r08 (29) | +1 | -11 | -4.7 | -25 | +29 | +2.2 | FAIL on C25 (noise) |
Baseline S AUC: C25 .81737 / .81758 (s0/s1); C23 .79675 / .79644.
Seed noise: S itself differs by 21e-5 (C25) and 31e-5 (C23) between seeds; per-seed deltas range -25..+36, so none of the deltas is statistically distinguishable from 0 with 2 seeds. The honest reading is "neutral, possibly slightly positive", not a proven gain. Paired means over all 4 (cut x seed) runs: a_p90 +14, b_mean +4.5, c_r08 -1.5 (x1e-5).
(Run times varied 230-1400 s because of external CPU contention, not set size.)

## Verdict
Cutting the a_p90 set (108 further columns, 401 -> 293) passes the keep rule at both cuts (+.00006 C25, +.00023 C23, worst single run -.00011) and is the best candidate; b_mean (95) also passes (smaller margin, worst -.00019 at one C23 seed). Both only claim neutrality within seed noise; the value is a 27% smaller feature set (faster fits, less overfit surface). The c_r08 subset formally fails C25 by noise only (-4.7 mean, one seed -11), which shows the 2-seed test cannot resolve effects below ~.0001 — consistent with the SP/SPC results from v19cut being worth confirming with more seeds. Recommend: adopt a_p90 only after a 3rd/4th seed confirms at C25/C23 and then a quick check at the C24 cut; if conservative, use b_mean.
Caution: SP (102) and SPC (114) lost earlier; they were chosen by gain/duplicate rank, a_p90 by null-gain, so the sets differ but overlap is not checked here.

## Files
- drop_regex_a_p90_extra.txt / drop_regex_b_mean_extra.txt : regex of the extra cols only (use with S already dropped; match exactly 108 / 95 of the 401 post-S features)
- drop_regex_a_p90_plus_S.txt / drop_regex_b_mean_plus_S.txt : S regex OR extra regex (re.search against the 484-col set after DROP)
- null_table.json, imp_*.json, sets.json, val_log.jsonl, run_*.py, lib.py, analyze_null.py, chain.sh/log
