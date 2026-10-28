# v12 new-signal search (dw_v12sig) — findings

Harness: dw_lb/lib.py residual LightGBM on v12 logits (init_score=z), user-grouped 5-fold, event-bootstrap 95% CI of AUC(test)-AUC(control). C25 primary, C23 confirm. v12 val AUC: C25 0.81723, C23 0.79566.
Noise control (null event-noise feature): C25 -0.00006 [-0.00012,-0.000005]; C23 -0.00002 [-0.00008,+0.00005]. Floor about ±0.0001.

## Ranked results (delta AUC vs control, 95% CI)
| Group | C25 | C23 |
|---|---|---|
| tmp: per-dim taste products (13) | +0.00019 [+0.00002,+0.00033] | +0.00028 [+0.00012,+0.00044] |
| tmprod: products + item coords (25) | +0.00017 [0,+0.00034] | +0.00035 [+0.00014,+0.00051] |
| tm: all taste-metric (30) | +0.00016 [-0.00002,+0.00032] | +0.00036 [+0.00017,+0.00053] |
| all (68) | +0.00015 [-0.00004,+0.00030] | +0.00044 [+0.00020,+0.00064] |
| tmi: item coords only (12) | +0.00002 [-0.00015,+0.00016] | +0.00014 [-0.00003,+0.00030] |
| tmbil: learned bilinear (5) | +0.00004 [-0.00008,+0.00016] | not run |
| cl: conditional-logit item strength (15) | +0.00002 [-0.00013,+0.00016] | -0.00001 [-0.00014,+0.00012] |
| sq: transitions/Markov (12) | +0.00002 [-0.00012,+0.00017] | not run |
| sigs all (38) | +0.00001 [-0.00015,+0.00015] | not run |
| rp: repeat propensity (4) | -0.00002 [-0.00011,+0.00007] | not run |
| lc: item lifecycle (7) | -0.00009 [-0.00020,+0.00001] | not run |

## Disagreement profile v12 vs p33 (no labels; t1_disagree.py, t1.log)
- Spearman 0.929, normal-score Pearson 0.935, residual sd 0.354; residual 15% event-level, 85% within-event.
- Spread across segments. Lowest agreement for no-history users (corr 0.47, tiny share). p33 ranks candidates already in the user's own history about +0.22 sd higher.
- Persistent ITEM-LEVEL residual: split-half reliability 0.36 (true item effect sd about 0.086), nearly uncorrelated (~0.07) with item CTR, popularity, lifecycle and CL strength. Hypothesis: p33 uses item identity / embedding (taste-region) information that v12 lacks. Matches the only winner: taste-dimension products from impression-space SVD.
- Event-level residual is 24% explainable from existing features (mostly SVD-family); within-event 7%.

## Task 4 (no/short history)
Not run as an experiment: no-history rows are 1.3% of eval (6,650), so even a big lift there caps at about +0.0005 global AUC.
## Task 5 (slate/position)
No signal. Click rate by file position is flat (chi2 p=0.84), rows inside an event are not sorted, rowid is a hash. Nothing to exploit, nothing leaky to worry about.

## Verdict
Nothing reaches +0.001. The only signal above noise is per-dimension user×item taste products (tm_prod0-11 + tm_prod_tail): about +0.0002 (C25, CI touches 0) and +0.0003 (C23, CI clear of 0), confirmed on both cuts, in the same direction. Item strength, lifecycle, transitions, repeat propensity, the learned bilinear metric and item-only coordinates are at the noise floor. Expect a small gain (~+0.0002–0.0004) if tm_prod is added as extra stacker features; probably worth it only if cheap. Likely v12 and p33 already capture the same sequence/item-strength information.

## Leak safety / tests
Features use only train events before the cutoff plus each row's own history and slate; no eval aggregates (c27 files are built from train events only). test_sigs.py (run, passes): taste_metric recomputed for val25 matches cache within 1e-3; item_strength is unchanged when post-cutoff labels are flipped.

## Files and reuse
- Code: sigs.py (extend_eval, item_strength, item_lifecycle, sequence, all_new), sigs2.py (imp_svd, learn_bilinear, taste_metric — `from sigs2 import taste_metric`; keep columns tm_prod*), run_groups.py (`run_groups.py <cuts> <groups>`), test_sigs.py, t1_disagree.py, t1b_item.py, t1c.py, t5_slate.py.
- Features: cache/sig_c{23,25,27}.parquet, cache/tm_c{23,25,27}.parquet (c27 = eval-time).
- Results: results.jsonl, run*.log, t1.log, t1_resid.parquet.
