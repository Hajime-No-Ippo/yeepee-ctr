# evclick — event-level P(any click) member (Variant A, CPU-only, nothing outside this folder written)
Verdict: **DEAD (fails pass rule).** Event model is real (event AUC .651/.675/.681 vs AUC Researcher .65-.66) but its information is already inside M; no blend gain.

## Build
- `build_ev.py`: per-event (10 rows, group by eid) aggregates of the top-70 gain features (mean/max/min/std; lse + 2nd-best for dot/logit/cos cols), mean of 20 hist_*, user cols (us_*, v2_u_ctr, tm_stale). Only the event's own slate + train-snapshot history; no eval-row aggregates. 346 features.
- `ev_train.py`: LightGBM binary, 600 rounds lr .03, 31 leaves, 4 threads, train on trn_tables(cut) events (C23 f1-2, C25 f1-4, C26 f2-5, eval f3-6), score val{cut}/eval events. Event AUC (P any click, 600 it): C23 .6507 (small train set, 2 f-tables), C25 .6746, C26 .6807. Top gain: sq_lse__mean, v2_u_ctr, us_ctr, sq_lse__max, coA_*_cos__lse.
- `ev_members.py`, `ev_fine.py`: item-level members, oof-row-aligned, blend vs eq4+graph0312 (C23 .79979 / C25 .82025 / C26 .82865).

## Results (AUC; gains in 1e-4 over base at best weight, grid .002-.4; HO = weight fit on one cut, scored on other)
| member | AUC C23/25/26 | rho vs eq4 | oracle gain | held-out |
|---|---|---|---|---|
| A broadcast rank(P_ev) | .560/.568/.571 | .08-.10 | 0 (<=+0.01) | 0 |
| B rank(P_ev)*rank(M) | .7225/.7397/.7445 | .73 | 0 | 0 |
| B rank(P_ev)*rank(eq4) | .7198/.7373/.7425 | .72-.73 | 0 | 0 |
| C logit: M+1*logit(P_ev) | .781/.801/.807 | .92 | 0 | 0 |
| C logit a=.5 | .794/.814/.822 | .975 | +.02/0/0 | -0.08, -0.22 |
| fine: A w=.002/.005/.01 | | | +.01 / -.03 / -.09 (C23/25/26 at .002) | all <= 0 |
| fine: C a=.05 vs M alone | | | +.26 / +.25 / -.71 | |
| fine: residual R a=.1 vs M alone (logit P_ev - M-implied any) | | | +.40 / +.44 / -.39 | |
corr(P_ev, M's implied P(any click)=1-prod(1-p)) = .894/.925/.931 -> M already encodes the event-level level (sq_*, v2_u_ctr, us_ctr are in the 484 set).
Max positive seen: +0.4e-4 vs M alone (not vs the full eq4+graph base), sign flips on C26. Pass rule (>=+1e-4 on both C23 and C25 vs eq4+graph, held-out) not met by any form.

## Not done (because dead)
Eval array / rho vs EN, S1 / binormal break-even were skipped: with no local gain there is nothing for them to qualify. `evpred_c27.npz` (eval event P, cut 27 training f3-6) exists if anyone wants it.
Caveat: only one LGBM config, one seed; no AUC-Researcher two-tower lse feature (its ingredients are already in sq_lse / tB_dot lse here).
