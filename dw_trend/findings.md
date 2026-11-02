# dw_trend — reconstructed "trend family" (趋势族) test  (2026-11-02)
Verdict: **NO group reaches the +0.0003 bar at either cut; nothing gains over the noise control.** Do not build a file / do not add to dwbig.

## Setup
- Base logits (local best): `.4*z + .6*M` (`base.py`): M = 8-seed mean of v21seq ctl6 var A (logit of col 3), z = v19 stack (C25) / dwbig (C23). AUC C25 .81907, C23 .79889 (≈.0001 below Grok Bot's rank-based .81921/.79903; row order = v10/v12 val{c}.parquet, asserted).
- Harness: dw_lb/lib.py residual LightGBM (init_score = base, GroupKFold by user, 5 folds, 2 seeds, 2 threads). control = base-context cols only; test = + group; **noise = same columns permuted across events**. CI = event bootstrap (200) of test-control. Raw output: `results_trend.jsonl`, `r25.log r23.log rtt25.log rtt23.log`.
- Eval-time rule change (Eric 02/11): TRANSDUCTIVE (TT_*) group added and reported separately.

## Features (trend.py, tt_trend.py; caches in cache/ for c23, c25, c27 — columns per parquet, joined on rowid)
Train-only, 13 raw (+ within-event demeaned copies for candidate-varying ones => 22 cols in "all"):
- item at cutoff X (last 1d vs prior 2d, 3 bins only because C23 has just 3 days of impressions): it_tr_ratio, it_tr_slope, it_imp_gr, it_clk_gr (dedup click log incl. history clicks), it_hrs_clk
- user own history: u_rate_tr (24h vs prior 72h clicks), u_gap_tr (newest-half vs oldest-half click gap), u_overdue
- taste drift (p33 12-d PPMI-SVD space, newest-5 vs older clicks, >=8 clicks): tx_cos_rec, tx_cos_old, tx_dcos, tx_drift, tx_proj
TRANSDUCTIVE (TT_, unlabelled val-fold rows: slates + timestamps; for C27 the eval rows): TT_imp_ratio (val-window item impressions vs train last-1d rate), TT_imp_loc (+-6h local burst), TT_slate_pop (event-level), TT_coimp_mean/max (PMI of candidate with last-5 history items in the val co-impression matrix).

## Results (delta vs control, 95% event-bootstrap CI; noise-permutation delta in brackets)
| group (cols) | C25 | C23 |
|---|---|---|
| item (10) | -.00005 [-.00018,+.00008] (noise 0.0) | +.00005 [-.00008,+.00016] (noise +.0001) |
| user (3) | +.00004 [-.00003,+.00012] (noise +.00008) | +.00016 [+.00007,+.00026] (noise +.00008) |
| taste drift (9) | +.00001 [-.00013,+.00011] (noise -.00016) | +.00015 [+.00001,+.00030] (noise +.0001) |
| all train-only (22) | +.00002 [-.00012,+.00015] (noise -.00007) | +.00012 [-.00005,+.00027] (noise +.00023) |
| **TT group (9)** | +.00003 [-.00011,+.00015] (noise -.00005) | +.00002 [-.0001,+.00012] (noise +.00008) |
| **TT + all (31)** | +.00002 [-.00014,+.00015] (noise -.00005) | +.00019 [+.00004,+.00036] (noise +.00025) |
Bar +.0003 at both cuts: **fails everywhere** (best C25 +.00004, best C23 +.00019, and C23's gains are no larger than the permuted-noise control; the C23 noise control itself is +.0001..+.00025, so the harness has ~.0002 slop at C23).
Note control is ~-.0001 (C25) / -.0003 (C23) below raw base (residual-LGB overhead), so "vs base" is negative for every group (see d_vs_base in jsonl).

## Why: univariate within-event AUC (uni.py/uni.log, events with a click, demeaned within slate)
- Item trends have **zero** within-event signal: it_tr_ratio .496/.497, it_tr_slope .501, it_imp_gr .499, it_clk_gr .501, it_hrs_clk .479/.483 (C25/C23). Item CTR is stationary at 1-3 day scale (<=1 click/item/day => estimates are mostly noise; consistent with the earlier staleness negative result). Item-level trends are NOT the 趋势族 gain, or at least are not recoverable here.
- Taste-drift: tx_cos_rec .774, tx_cos_old .790 (older centroid is slightly *better*, i.e. no recency edge), tx_dcos .48 and tx_proj .48 -> drift direction carries nothing; confirms dw_p33 (recency weighting adds nothing).
- TT_imp_ratio/loc .49-.50 (val-window popularity carries no label signal); TT_coimp_mean/max AUC .37/.38 (= .63 inverted: strongly anti-correlated, but already captured by existing impression-space/co-impression features: residual gain ~0).
- User trends (u_*) are event-level: they can only help the between-event part. C23 +.00016 CI [.00007,.00026] but noise control +.00008 and C25 +.00004: not reproducible.
=> The teammate's +.00075 was measured on a 484-col base of theirs; on our stronger ctl6+z base nothing trend-like survives. Most plausible: their trend family is user/time-level (event-level) features that our base already covers via v2_*/ev_*/dw_prop families, or a different base.

## Leak tests (test_trend.py, passes)
Flipping every label at/after X, and ALL pre-cutoff labels for TT features: TT identical (they use no labels). Train-only features identical when only post-cutoff labels flip; they DO change (it_tr_ratio, it_tr_slope, it_clk_gr, it_hrs_clk) when pre-cutoff labels are randomised (power check). Train-only item stats ignore post-cutoff impressions.
Leak-safety of TT group: uses only slates/timestamps of the fold's own first-after-cutoff events (unlabelled), mirrors what eval offers.

## Not done
dwbig LightGBM add-on check: skipped since no group passes the harness (to run: append cache/trend_c{c}.parquet + tt_c{c}.parquet cols joined on rowid to the feat dir, same --drop recipe as v19/xlgb17.py). No uploads, no files built.
## Files
trend.py (leak-safe builders, `python trend.py 23,25,27`), tt_trend.py (TRANSDUCTIVE), base.py, run_trend.py, uni.py, test_trend.py, cache/{trend,tt}_c{23,25,27}.parquet, results_trend.jsonl, logs.
