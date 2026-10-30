# v18classic — freshness / audience / user-time / typed-CTR families on top of v17cooc A (train-only)

STATUS: NEGATIVE. No family gives a consistent gain; no full blend, candidate or probe built (criteria not met). Nothing uploaded, nothing copied to Mac.

## Features (all strictly pre-timestamp; snapshot method as v12/v17cooc A; builder mkv18.py, ~4-5 s/table, 40 s for all 10 tables)
- feat_fr (14): item age since first seen, time since last impression / last click by anyone (hours + window-relative), within-slate rank/z/dm.
- feat_au (16): mean/std of past clickers' user CTR, history length, staleness; this user's stat minus clicker mean (+z), slate dm.
- feat_uc (11): smoothed (a=3) share of user's past events/clicks in current time_bucket/day_type; hour-of-day distance to typical hour.
- feat_ty (15): item CTR (m=20) among heavy/light clickers and new/returning users, value for this user's type, lift over overall item CTR.
Leak test (test_leak18.py; rebuild==stored, label flip, corrupt non-target post-cutoff events): passed for val25 and val23 (logs/test_leak18.out).

## Single-feature screen (val25, within-event AUC; logs/screen1_val25.json)
fr_age_h .5167, fr_gap_imp_h .4865, fr_gap_clk_h .4788, fr_gap_clk_rel .4764; au_nclk_item .5315, au_dist .4833, au_ctr_d .4940; ty_ctr_ret .5288, ty_ctr_n_dm .5289, ty_ctr_h .5278, ty_lift_n .5164; uc_* = .5000 (event-constant).

## Freshness vs popularity (partial_fr.py, val25)
R^2 of freshness features on popularity features: age .190, gap_imp .109, gap_clk .367, gap_clk_rel .319. Residual within-event AUC: age .5015, age_norm .5014, gap_imp .4982, gap_clk .5040, gap_clk_rel .5033. Logistic within-event AUC (2-fold CV by event): popularity .5377, +freshness .5379 (+.0002). Freshness is almost entirely a popularity proxy.

## dwbig LightGBM (recipe of v17cooc A: dwbig + feat_Ap, 1000 rounds), delta vs stored v17cooc A
Baselines: C25 .81719, C23 .79631, C26 .82599. Seed noise ~ +-.00006.
| config | C25 | dC25 | C23 | dC23 | C26 |
| fr  | .81721 | +.00002 | .79630 | -.00001 | not run |
| au  | .81707 | -.00012 | not run | | not run |
| uc  | .81719 | +.00000 | not run | | not run |
| ty  | .81712 | -.00007 | not run | | not run |
| all | .81683 | -.00036 | not run | | not run |
Runtimes: 4.5-4.8 min (fr, uc, all), 8.7-10.5 min (ty, au; box load) per 1000-round fit.
I stopped the screening loop after C25 (all 5) + C23 fr, since no family beat noise at C25 and the combination is worse (-.00036); remaining C23/C26 runs would not change the decision. Raw: ens/xlog.jsonl.

## Decision
Kept set = empty. Full blend, submission_v18_candidate and p33 probe NOT built (needs >= +.0002 on C25). v17cooc A (.81802 blend) remains the best.
