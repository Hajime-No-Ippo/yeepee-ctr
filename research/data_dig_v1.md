# Raw-data dig v1: looking for missed signal (17:05 IST, 23 Oct 2026)

**Setup:**
- User sample: `users.hash(seed=7) % 100 < 8`, which gives 4,269 of 53,003 users, 53,669 train events and 536,690 rows. Of those rows, 35,230 are val25 rows (3,523 events) and 27,990 are val26 rows (2,799 events).
- Ran under nice 19 with POLARS/OMP/MKL/OPENBLAS threads set to 2. Peak RAM was about 2 GB.
- Only train data and the val25/26 labels were used. For eval, I read only the row's own columns (the timestamp range and the CSV head).
- Scripts are in `research/data_dig_v1_scripts/` (copied from /tmp/dig). No pipeline files were touched.
- **Deviation, disclosed:** the item-embedding test (F1) fits item vectors on *all* users' pre-cutoff histories. I used `v2work/train_events.parquet` `h_items`, not the CSV. Each fit takes about 10 s at 2 threads. An 8% user sample cannot fit 50k item vectors. Evaluation is still on the sampled users' val25/26 rows only.
- **"resid" = incremental test.**
  - Take the base model's val score (`scr_base_c25/26.npy`; on the sample it scores val25 0.7729 and val26 0.7823 AUC) and use it as init_score.
  - Fit a tiny LGBM (7 leaves, 150 rounds) on the candidate feature only.
  - Use 5-fold CV grouped by user on the sample's val rows and average 2–3 seeds.
  - Report ΔAUC vs base.
  - Noise control (3 Gaussian columns): −0.0012 / +0.0004. Treat |Δ| < ~0.0015 as zero.

## Ranked findings

### F1. The data has a ~12-dimensional latent factor structure, and our SVD features recover it badly. This is the one real finding.
- **Spectrum.** Take the user×item click matrix: each user's latest history before cutoff C, binary, columns scaled by 1/sqrt(pop+5). Its singular values are one popularity component (3.34), then a block of about 12 components at 1.94–2.11, then a decay down to a flat noise floor at about 1.70. That pattern is a planted rank-~12 model.
  - KMeans silhouette is about 0.07, so the latent is continuous, not discrete topics.
- **Feature.** For each candidate, compute the cosine between its vector and the centroid of the event's own history vectors.
  - Item vectors are components 2..13 of an exact `svds` (ARPACK) fit, scaled by the singular values.
  - The matrix rows are each user's last 30–40 clicks before C (union over the train histories).
  - Columns are scaled by 1/sqrt(pop+5).
  - Vectors are fitted at C=J25 and reused for val26, which makes val26 one day stale and therefore conservative.

| feature (sample) | glob AUC v25 / v26 | within-event v25 / v26 | resid Δ v25 / v26 |
|---|---|---|---|
| **new: last30, k12, comps 2..13** | **0.772 / 0.768** | **0.773 / 0.770** | **+0.0133 / +0.0083** |
| new: last40, k12 | 0.771 / 0.770 | 0.775 / 0.771 | +0.0130 / +0.0091 |
| new: last40 k12, raw + within-event z + dmax | – | – | +0.0139 / +0.0091 |
| new: last20, k12 | 0.761 / 0.757 | 0.761 / 0.761 | +0.0102 / +0.0057 |
| existing best single: x_con05_k32 | 0.724 / 0.738 | 0.725 / 0.743 | (in base) |
| existing v2_svd16_cos | 0.708 / 0.725 | 0.712 / 0.725 | (in base) |
| existing sv_k16_ucos | 0.685 / 0.697 | 0.693 / 0.706 | (in base) |

- **Coverage.** The nearest existing feature is v2_svd16_cos, with Spearman |ρ| = 0.63. This one feature roughly matches the whole 325-feature base model on the sample (0.772 vs 0.773), and it adds about +0.009 to +0.014 on top of it.
- **Why the existing SVD features miss it** (ablation on the same last-30 data, within-event v25 / v26):
  1. **Randomized SVD does not converge on this flat spectrum.**
     - `randomized_svd(k=12, n_iter=5)`: 0.711 / 0.709.
     - `n_iter=20` or exact `svds`: 0.773 / 0.770.
     - feats.py, v3/features.py, extra.py and extra2.py all use n_iter 4–5.
  2. **IDF times row-L2 weighting is much worse at low k.**
     - IDF+rownorm, exact svds: k12 = 0.63–0.65, k16 = 0.69, k32 = 0.72.
     - Existing-style setup (k64 with n_iter 4, first 16 components): 0.68.
  3. **k matters a lot.** Every component past the signal block adds noise with equal weight in the cosine.
     - k = 4 / 8 / 12 / 13 / 16 / 20 / 32 / 64 / 128 gives 0.57 / 0.69 / 0.76 / 0.76 / 0.75 / 0.74 / 0.72 / 0.68 / 0.66 (last-20 data).
     - Dropping the first (popularity) component gives a small +0.002.
  4. **Count weighting hurts.** Repeated clicks as counts give 0.62 vs 0.72 binary at k32. Use binary.
  5. **Column scaling matters.** alpha 0.5 is best: alpha 0 gives 0.66 and alpha 1.0 gives 0.71–0.74.
- **Caveat: the choice of training rows for the matrix is fragile, and I could not explain it.**
  - Last 20 / 30 / 40 / 80 clicks per user give 0.76 / 0.77 / 0.775 / 0.765.
  - Clicks since J13 only give 0.59, and window-only clicks (J20+) give 0.52. Warm-up-only clicks give 0.75.
  - A union of all clicks (svdvar4 "full") gave 0.62–0.71.
  - Yet pairwise coherence in the good space is the same for warm-up and window clicks (0.258 vs 0.261). So this looks like an estimation or normalisation artefact, not a different process.
  - **Implication:** a properly fitted low-rank model is likely to beat my quick SVD. Candidates: implicit ALS or weighted MF at d≈12–16, or a d≈12 two-tower/history-centroid model trained with an in-event softmax on train events.
- **Leak check:**
  - Vectors use only train histories with event ts < C.
  - The centroid uses only the row's own history, the same as the existing features.
  - For eval, fit on train histories only at C=J27, with no eval histories in the matrix. That complies with Eric's rule.
- **Expected gain:** about +0.008 to +0.014 global AUC on top of the base, from one feature family. That figure is a sample estimate; confirm it with a full screen.
- **Suggested next step (not done here; it needs pipeline work):**
  - Add `lr12_cos` (+ `_z`, `_dmax`) built exactly as above at each fold cutoff, then screen it.
  - Then try the same k≈12 space for the x_con05 contrast (click centroid − 0.5 × non-click impression centroid) and for the ucos fold-in. Their k32–128 versions are currently the top features.

### F2. Event-level click probability: nothing beyond user CTR. Already covered.
Event-level AUC on val events (sample), v25 / v26:

| feature | AUC v25 / v26 |
|---|---|
| user's prior CTR (shrunk) | 0.613 / 0.619 |
| prev event clicked | 0.548 / 0.548 |
| hist length | 0.544 / 0.545 |
| click rate in history | 0.548 / 0.547 |
| median gap between history clicks | 0.536 / 0.535 |
| time since last history click | 0.528 / 0.532 |
| warm-up share of history | 0.522 / 0.533 |
| time since previous event | 0.509 / 0.505 |
| hour | 0.508 / 0.518 |
| minute | 0.503 / 0.504 |
| user hash | 0.518 / 0.517 |
| event hash | 0.505 / 0.508 |

- A GBM on 12 event features gives 0.610 / 0.620, the same as prior CTR alone. The base model's event AUC is 0.612 / 0.618. Event-level signal looks capped at about 0.62.
- prev_y has a little partial effect within CTR quintiles, but it is ρ = 0.999 with `s_clk_last1`.
- `since_hclick`, `since_prev_ev` and `gap_med` have ρ of 1.00, 1.00 and 0.92 with `us_since_clk`, `v2_u_since_ev` and `v2_h_gap_mean`.
- **Expected gain: 0.**

### F3. Candidate slot / order: no signal.
- Every event is a contiguous 10-row block.
- CTR by slot ranges 0.0625–0.0649 (n = 53.7k each, binomial SD ≈ 0.001).
- Slot single AUC: 0.504 / 0.505 global, 0.504 / 0.505 within.
- The correlation of slot with the item-ID rank inside the event is −0.001.
- **File order:**
  - The file is sorted by timestamp (100% monotone; Spearman 1.000).
  - Within a minute, event order has AUC 0.49999 vs click, and ρ ≈ 0 with event hash, user hash and history length.
  - Events per minute vs click: 0.4994.
- **Expected gain: 0.**

### F4. Within-event / slate structure: real structure, but already covered.
- **No duplicates and no repeated slates.** There are 0 duplicate items within an event and 0 identical slates in the sample.
- **Slates are personalised.**
  - P(candidate in the user's history) is 2.6% observed vs 0.05% expected under popularity sampling, about 50× higher.
  - Re-showing an item not in history happens at 0.90% vs 0.44% random.
  - CTR by in-history status and number of prior showings: in history 0.29–0.44, not in history 0.043–0.057. A re-shown item that was never clicked has normal CTR, 0.0555 vs 0.0573.
  - Events with k in-history candidates have click rates 0.607 / 0.724 / 0.768 / 0.81 for k = 0 / 1 / 2 / 3. This is covered by `v2_ev_n_inh`, `ev_hist_n_all_*`, `v2_inh_*`, `hist_*` and `v2_ui_noclk*`.
- **Slate similarity (my own SVD k32, resid Δ v25 / v26):**
  - The clicked candidate is slightly *more* similar to the other 9 (it is not an odd one out): within 0.527 / 0.538, resid −0.001 / −0.001.
  - Co-slate (impression) embedding similarity: within 0.530 / 0.534, resid −0.003 / −0.002.
  - Similarity to the user's past non-clicked impressions: within 0.54–0.55, resid −0.002 / −0.001 to −0.002.
  - All of these correlate with popularity (ρ 0.35–0.48 with `it_imp_rate`).
- **Item popularity is weak.**
  - Between-item CTR variance is only 12% of observed variance; the rest is binomial noise. Split-half item CTR correlation is 0.11.
  - Pre-C impressions, clicks and smoothed CTR give single AUC 0.52–0.53.
- **Expected gain: 0.** Beyond F1, the useful signal is user–item affinity.

### F5. ID patterns: random.
- IDs are `row_`/`evt_`/`usr_`/`itm_` followed by 32 lowercase hex characters, which looks like an md5-style hash.
- Spearman correlations are all |ρ| < 0.008:
  - item hex vs impressions, CTR, first impression ts and first history ts;
  - user hex vs number of events and user CTR;
  - event hex vs ts (−0.002).
- **First-seen order:**
  - All 50,000 items exist from the start of the warm-up. The first history click falls at a median of day 1.9 and a 99th percentile of day 12.7 after Jan 6, and it correlates with popularity at −0.42.
  - Every item clicked in a history was also impressed in train.
- **Expected gain: 0.**

## (a) Exact raw schema and how it maps to features
- **Columns:**
  - `rowid`, `event_id`, `user_id`, `item_id`: string, prefix plus 32 hex characters. prep.py `hexid` keeps the first 15 hex characters as int64. The sample shows no collisions: 50,000 distinct items, matching the catalogue size.
  - `timestamp`: int Unix seconds, always a whole minute (100%). Train runs Jan 20 00:00 – Jan 26 23:59 UTC; eval runs Jan 27 00:00 – Feb 2 23:56 UTC. No eval event precedes train, so there is no future-history leak.
  - `time_bucket`: equals (ts % 86400) // 21600 exactly.
  - `day_type`: 1 for Saturday and Sunday. Both are deterministic functions of ts.
  - `history_item_ids` / `history_timestamps`: JSON string arrays inside quoted CSV.
    - History is the last ≤ 20 clicks, ascending; 68% of sampled events have exactly 20.
    - Ties are possible (minimum diff 0). Timestamps are whole minutes.
    - 76% of histories contain a repeated item.
    - A click's ts equals its event's ts: 99.8% of clicked events show up as the last history entry of the user's next event, vs 0.1% for non-clicked events.
    - There are no clicks outside events in the window: 0 history clicks fall strictly between consecutive events.
  - `clicked`: at most one per event; train only.
- **There is no unused raw field.**
  - ts, tb and dt feed the time features (`s_*`, `us_since_*`, `v2_u_since_ev`, ...).
  - The history lists feed `hist_*`, `v2_h_*`, `v2_inh_*`, `co_*`, `sv_*`, `x_*`, `v2_svd*`, `r_l*`, `ev_hist_*`.
  - Item and user IDs feed the `it_*`/`v2_i_*` and `us_*`/`v2_u_*` aggregates.
  - The ID strings themselves carry nothing (F5).
- The only "crudely used" input is the click log's low-rank structure (F1).

## Bottom line
Only one new thing turned up: F1. The existing SVD/embedding features capture the planted ~12-dimensional user–item factor structure poorly. The causes are non-converged randomized SVD, IDF+row-norm weighting, and a k that is too large. A correctly fitted k≈12 history-centroid cosine is the best single feature we have, at about 0.77 within-event, and on the sample it adds about +0.009 to +0.014 global AUC over the base. A better-fitted d≈12 factor model could add more, but that is unmeasured. Everything else (position, ordering, timing, event-level, ID patterns, slate structure) is null or already covered.
