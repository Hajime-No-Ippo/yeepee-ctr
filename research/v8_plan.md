# v8 plan: from v7 (val 0.81164 / 0.82099) toward 0.82–0.83 public

Inputs: `research/v7_error_analysis.md` (bottlenecks), the v7 screens, the v7 lite measurements, and Code Buddy's v5 step A/C.
Rules unchanged: train data plus each eval row's own history only, nothing uploaded without Eric, 4 threads.
Runtime reference: the measured v7 lite_nox2 4-core end-to-end time is in `lite_summary.md`.

## Bottlenecks being targeted

- **B1 (≈0.017 weighted deficit):** low-activity users, activity deciles d0–d2. AUC 0.72–0.79; 33.5% of eval.
- **B2 (≈0.008 plus an unmeasured tail):** long-gap and stale users. 60% of eval is ≥ 1 d since last activity, and 16.8% of eval is ≥ 2 d past the cutoff, which no val set covers. Reweighting to eval's gap mix alone costs −0.002 to −0.005.
- **B3 (≈0.005):** empty-history and cold users (AUC 0.54 / 0.75, 3.4% of eval).
- **Structural:** two-thirds of the error is within-event ranking (perfect-ordering oracle +0.15 vs +0.07 for event labels).
- **Validation gap:** no Monday or weekday val, though eval is 66.6% Monday.

## Step 0: fix validation first (cheap, needed to measure B2 and Monday)

- **Add val23 to v2work** (cutoff Jan23, first event per user until Jan27). It gives staleness 0–4 d and Thu/Fri weekday rows; Code Buddy's v5 already uses it.
- **Gate every v8 change on val25 + val26 + val23,** plus the eval-mix-reweighted AUC (gap mix from err2.py).
- **Cost:** one feature build per stage for one extra split, about 4–5 min. No submission-runtime impact.

## Ranked experiments

| # | Experiment | Targets | Expected gain (val) | Dev cost | Runtime impact |
|---|---|---|---|---|---|
| 1 | **Port Code Buddy's step C PPMI-SVD item embeddings (item2vec-style, `v5/feat_pp`) onto v7.** Copy read-only; don't rebuild a second item2vec. Screen on v7 std+gap. On v5 it gave +0.0064 / +0.0044 / +0.0114 (val25/26/23); v7's two-tower may overlap, so expect less. | within-event, B1 | **+0.002 to +0.004** | 30 min | +1–2 min (ppmi build) |
| 2 | **Stale-like training targets plus staleness weighting.** Add "first event per user in [C, Jan27)" folds (W=None) so training sees 0–6 d gaps. Weight rows to eval's gap/stale mix, try weights 1.5–2 on ≥ 1 d rows. Add the features `tm_stale × us_since_ev`, `us_since_ev / user median gap` and `last-activity bucket`. | B2 | **+0.002 to +0.005 on eval-mix AUC** (raw val may not move) | 1.5 h | +0–2 min (bigger folds; offset with lite negfrac) |
| 3 | **Low-activity / cold-user tower.** Item-side contrast and item-to-item co-click scores against the few history items (max/mean over history), plus popularity-by-cluster priors. Optionally a separate shallow LGBM for us_nclk_all ≤ d2 and empty users, blended by segment. | B1, B3 | +0.001 to +0.003 | 2 h | +1 min |
| 4 | **Small NN blend member** (two-tower plus listwise softmax over the 10 candidates, on v7 features plus id embeddings). Rank-average or logit blend with LGBM. Choice-aware training directly attacks the within-event part. | within-event | +0.002 to +0.004 (two-tower features already gave +0.01; a listwise head is the next step) | 2–3 h | +2–3 min on CPU (6 epochs; cap at 1M rows) |
| 5 | **LGBM tuning sweep on the v7 feature set** (25% screen, then full): leaves 63–255, min_data 100–800, ff 0.4–0.7, l2 10–60, lr 0.03 with early stopping, plus `lambdarank`/`rank_xendcg` per event as a blend member. | general, within-event | +0.001 to +0.002 | 1 h of box time | ±0 to +2 min |
| 6 | **Day-of-week / weekday features**: `day_type` and `time_bucket` interactions with user and item CTR by weekday vs weekend (train only), and a weekday flag on the target. Validate on val23 (Thu/Fri) vs val25/26 (weekend). | Monday shift | +0 to +0.001 val; it's a risk hedge for 66.6% Monday | 45 min | negligible |
| 7 | **xA clean columns (`feat_xAn`)**: +0.0003 on v7. Include only if it's free after #1. | – | +0.0003 | done | +1 min (extra.py NEW) |

**Budget path to 0.82–0.83 public:**
- v7 public is unknown, because nothing was uploaded, but val26 is 0.821.
- #1 + #2 + #4 is about +0.006 to +0.012 on val (not additive; expect about 60%). That's the realistic route to the top of the 0.82–0.83 range.
- #5 and #3 are polish.

## Runtime guard (15 min laptop budget)

- Build v8 on the **lite_nox2** recipe:
  - 1 base seed + slow
  - 50% negatives (all clicks kept)
  - folds f3–f6
  - max_bin 63
  - extra2.py dropped
- Its exact 4-core time is in `lite_summary.md`. Every new stage gets timed with `v2work/timed_run.py` (JSON lines in `v2work/out/timings.jsonl`).
- Admission rule: a v8 addition must give ≥ +0.0005 on val25/26/23 per extra minute of 4-core wall time.

## Order of work

0. Build val23 for v2work.
1. Port PPMI (#1) and screen.
2. Stale folds/weighting (#2) on val23.
3. NN listwise member (#4).
4. Tuning sweep (#5).
5. Day-of-week features (#6) and the cold tower (#3) if time allows.
6. Rebuild v8 lite, time it, validate, and hand it to Eric for upload approval.

## Additions (parent steering, 23:32 IST)

### A. Adversarial validation (DIAGNOSTIC ONLY)

Train-fold rows (f3..f6, the c27 training rows) vs eval rows, 150k sampled from each, on the v7 feature matrix. LGBM 3-fold CV; script `research/v7_err_scripts/adv.py`, output `adv.json`.
It is **not** used to create features. Any drop list below is a **proposal for Eric to approve**, because it looks at eval rows (unlabelled, feature distributions only).
Results: see "Adversarial validation results" at the end of this file.

### B. Diverse CPU-cheap blend members

| Member | How | Expected | Runtime (4 cores) |
|---|---|---|---|
| **GBDT+LR** | Leaf indices of the v7 base model (400 trees × 63 leaves, one-hot, sparse) → L2 logistic regression (liblinear/saga) on the training folds. Blend with the base/slow logits. | +0.0005 to +0.0015 (corr with LGBM about 0.97; most useful within-event) | about 1–2 min (predict leaves is cheap; LR on 750k × 25k sparse) |
| **FM / FFM** | No CPU FM library installs cleanly here: xlearn is sdist-only from 2019 and won't build on py3.13; pyfm/fastFM are absent. Implement a **torch FM** (already installed, 2.14 CPU). Fields: user, item, hist items (bag), time_bucket, day_type, item-popularity bin, user-activity bin, plus about 20 dense top-gain features. 2–3 epochs, embedding dim 16. FFM is the same code with per-field embeddings and is slower; try FM first. | +0.001 to +0.003 (an id-level member LGBM doesn't have; overlaps the two-tower features) | about 2–3 min with 4 threads at 750k rows |
| **Blend method** | Compare logit averaging (current), per-event rank-normalised averaging (rank within the event / 10 plus a small event-level term), and global rank-averaging. AUC is global, so a pure per-event rank throws away between-event signal. Use `logit_mean + w × within-event z` and tune w on val25, checking on val26/val23. | +0.0003 to +0.001 | seconds |

### C. Cross-CTR audit

The data has **no item or user attributes**; the guideline lists only opaque ids.

| Column | Cardinality |
|---|---|
| user_id | 54,456 users (52,939 in eval) |
| item_id | 50,000 items |
| time_bucket | 4 |
| day_type | 2 |
| history items | up to 20 per event |

Derived categoricals: hour of day (24), weekday (7), user cluster (KMeans 64/256, train-only), item-popularity bin, user-activity bin.

Already covered:
- item CTR (all / 24 h / 6 h, trend)
- item × time_bucket CTR (`v2_i_ctr_tb`, diff, rank)
- user CTR
- user-cluster × item CTR (`x_clctr64/256`)
- user × item repeat impressions (`v2_ui_imp`, since)
- co-click counts (`v2_co_cnt`)
- PPMI and SVD similarities

**Missing; proposed as past-only (ts < C, labels from the fitting part only) with Bayesian smoothing toward the parent CTR (m ≈ 20):**
1. item × day_type CTR, and item × weekday-vs-weekend lift (a Monday hedge; uses day_type)
2. item × hour-of-day CTR (24 bins, smoothed to item × time_bucket)
3. user × time_bucket and user × day_type CTR (smoothed to user CTR); user-cluster × time_bucket
4. user-cluster × item-popularity-bin CTR, a cold/low-activity prior (B1/B3)
5. last-history-item × candidate co-click CTR: the share of impressions of the candidate after clicking item h that were clicked (directional transition CTR; train events only)
6. item × user-activity-bin CTR (do light users click this item?), targeting B1
7. item × staleness-bucket CTR (impressions served ≥ 1 d after the user's last activity); B2

Expected +0.001 to +0.003 combined. One polars group-by stage costs about 1 min.

### D. Multi-window user stats

Present now:
- `us_nclk24` and `us_nclk_win`
- `s_nev24h` and `s_clk24h` (in extra2.py, which **lite_nox2 drops**)
- `v2_up_1d`
- since-last-event and since-last-click

Missing:
- last 1 h / 7 d impressions, clicks and CTR
- user impressions in the last 1 d (only s_nev24h)
- the trend ratio 1 d / 7 d

Train impressions plus own-history clicks only. Since extra2.py is dropped in lite, port a small `uwin.py` (1 h / 1 d / 7 d × imps / clicks / CTR / ratios, about 12 cols, about 30 s). Expected +0.0005 to +0.0015, mostly on B2 (it separates "gone quiet" from "always sparse").

## Combined ranking (all candidates)

| Rank | Item | Expected | +min (4-core) |
|---|---|---|---|
| 0 | val23 + eval-mix AUC gate | enables the rest | 0 |
| 1 | PPMI item2vec port (step C) | +0.002 to +0.004 | +1–2 |
| 2 | Stale folds, weighting and gap features | +0.002 to +0.005 (eval-mix) | +0–2 |
| 3 | Small NN listwise member (or torch FM) | +0.002 to +0.004 | +2–3 |
| 4 | Missing cross CTRs (C.1–C.7) | +0.001 to +0.003 | +1 |
| 5 | Cold / low-activity tower | +0.001 to +0.003 | +1 |
| 6 | LGBM tuning plus a lambdarank member | +0.001 to +0.002 | 0 to +2 |
| 7 | GBDT+LR member | +0.0005 to +0.0015 | +1–2 |
| 8 | Multi-window user stats (uwin) | +0.0005 to +0.0015 | +0.5 |
| 9 | Blend: logit + within-event z | +0.0003 to +0.001 | 0 |
| 10 | Day-of-week features | 0 to +0.001 (hedge) | 0 |
| 11 | xAn clean columns | +0.0003 | +1 |
| – | Adversarial-validation-driven drops | proposal only, Eric to approve | – |

## Adversarial validation results (diagnostic, run 23:58 IST)

150k rows from f3..f6 vs 150k eval rows, v7 matrix, LGBM 3-fold with 4 threads; 81 s.
- **Adversarial AUC = 1.000.** After removing the top 10 gain features it is still **0.9958**.
- Top drifting features (gain share / single-feature AUC):
  - ev_it_imp_mean 0.347 / 0.833
  - it_imp 0.128 / 0.684
  - ev_it_clk_dec2d_mean 0.119 / 0.507
  - v2_h_since_first 0.114 / 0.679
  - it_imp_rate 0.108 / 0.501
  - v2_ip_per_imp 0.034 / 0.728
  - ev_it_ctr24_mean 0.022
  - v2_i_imp_rate 0.019
  - v2_i_imp 0.014 / 0.684
  - it_ctr6 0.012
  - then ev_it_imp_max (0.752), it_clk (0.642), v2_u_since_ev (0.646), us_since_ev (0.637)
- **Interpretation:** this is almost entirely trivial *temporal* drift. Cumulative counts (item impressions and clicks, history-since-first) grow with the cutoff, so any day can be told apart from any other. The second cluster is staleness (`*_since_ev`), which matches the known eval shift (stale 33% vs 20%).
  - Single-feature AUCs of about 0.50 with high gain (it_imp_rate, the decayed clicks) mean the separation comes from interactions with the cumulative counts, not from the rates themselves.
- **Use:** none for features, as instructed. A per-day adversarial check (f6 vs f5) would be the right null to separate "normal day-to-day drift" from "eval-specific drift"; that's for the next session.
- **Proposal for Eric to approve (not applied):** replace or add cutoff-normalised versions of the raw cumulative counts:
  - it_imp, v2_i_imp, ev_it_imp_mean/max and it_clk divided by the total impressions in the window
  - or a log-ratio to the same-day global volume

  Then A/B on val25/26/23. Dropping raw counts outright is **not** proposed, since they carry popularity signal.
