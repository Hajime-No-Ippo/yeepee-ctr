# stale — stale-user-aware member (Variant A, CPU-only, writes only in this folder)
Verdict: **DEAD (fails pass rule; stopped after the cheap approximation).**

## What was and wasn't feasible
- A faithful version (user features recomputed with a 2-day gap before each training event, full top-80/490 features) needs the whole feature pipeline re-run per pseudo-cutoff; features in v9/feat_v3 f1..f6 are precomputed at fixed per-day snapshots (tm_stale <= 24h in every f table: 0% >1d, vs eval 33.4% >1d, val C23 30.4%, C25 19.9%, C26 0%). Not rebuildable in the time-box.
- That exact hypothesis was already tested by /workspace/huawei/staleness_test (lite features, rebuilt delay-matched B / B_w rows with 2-3 day gaps): stale-subset AUC unchanged for B, delta -0.0001 .. -0.0036 rw AUC.
- Approximation run here (`stale_train.py`): LightGBM on the top-80 gain features (lr .05, 63 leaves, 400 rounds, 4 threads), trained on trn_tables(cut) as in the base, 4 variants: base80 (plain), w (row weight 1+3*(tm_stale/1d)^2: emphasise the stalest training rows), wt (w + tm_stale as feature), t (tm_stale feature only).

## Results (val AUC; stale = tm_stale > 86400 s)
Reference M (existing): stale-subset AUC C23 .7765 vs fresh .8076; C25 .8056 vs .8223 (stale users are genuinely harder).
| variant | C23 all / fresh / stale | C25 all / fresh / stale | C26 all |
|---|---|---|---|
| base80 | .7949 / .8043 / .7722 | .8157 / .8191 / .8016 | .8237 |
| w | .7938 / .8030 / .7716 | .8150 / .8184 / .8005 | .8232 |
| wt | .7938 / .8038 / .7696 | .8147 / .8184 / .7996 | .8234 |
| t | .7947 / .8040 / .7723 | .8153 / .8186 / .8015 | .8237 |
Stale-subset AUC vs plain: -0.0006/+0.0001/-0.0026 (C23), -0.0011/-0.0020/-0.0001 (C25): no variant improves stale users.
Stale-subset AUC of eq4+graph base: C23 .7775, C25 .8065 (above every top-80 member).
Blend (weight on rank(member) in eq4+graph0312; rho vs eq4 .973-.983, vs M .965-.978): oracle gains <= +0.01e-4 on all cuts; held-out gains 0 to -0.13e-4. Stale-subset gain of adding the member: <= +0.1e-4.
Eval array, rho vs EN/S1: skipped (nothing to qualify). Caveat: top-80 approximation, one seed, no true 2-day-gap feature recompute; deltas are consistent with the earlier staleness_test null.
