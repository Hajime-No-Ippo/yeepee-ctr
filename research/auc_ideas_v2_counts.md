# AUC ideas v2: time-dependent count features (drift, horizon, trend, own context, time-aware TE)

Written 2026-10-23 15:14 IST by the executor (read-only on pipelines; the only write is this file).
Baseline: public 0.788757 / local val 0.7785. Hard rule (Eric): no aggregates over eval rows. Every idea below uses
train-only global tables (CLg / P_rows before cutoff C) plus each row's own history and its own 10 candidates.

## 0. Grounding checks run here (light polars, nice 19, train folds + val25/val26 only, no eval reads)

**A. Which model features drift (60k-row samples, KS vs val25, single-feature adversarial AUC f4 vs val25).**
- The ~1.0 adversarial AUC comes from absolute-time columns: tm_dt, v2_ts, v2_dt (KS 1.000), v2_u_last_ev (KS f4 0.73, adv 0.83), v2_up_last, v2_ui_last_imp, v2_uip_last.
  **None of these is in the 325-feature model** (checked against /tmp/imp_s0.json).
- Among features that *are* in the model, the biggest shifts are cumulative counts and "time since" values that grow with C:

| feature | KS f1 vs val25 | KS f4 vs val25 | adv AUC f4 vs val25 | median f1 → f4 → val25 (→ val26) |
|---|---|---|---|---|
| ev_it_imp_mean | 0.996 | 0.296 | 0.706 | 28.9 → 114.9 → 143.6 → 172.5 |
| v2_ip_per_imp | 0.850 | 0.194 | 0.625 | 0.51 → 0.205 → 0.177 |
| it_imp (= v2_i_imp) | 0.764 | 0.137 | 0.595 | 22 → 87 → 109 → 131 |
| v2_ui_since_imp | 0.692 | 0.160 | 0.598 | 63.8k → 159k → 211k s |
| it_clk | 0.586 | 0.097 | 0.569 | 1 → 6 → 7 → 8 |
| v2_h_since_first | 0.535 | 0.242 | 0.602 | 1.12M → 1.29M → 1.41M s |
| us_nev | 0.579 | 0.094 | — | 2 → 6 → 7 → 10 |
| it_imp24 / it_imp_rate / it_ctr | ~0 | ~0 | — | 22 / 22 / 0.062 (stationary) |

  So the claim is confirmed for the *model's* features: cumulative counts grow roughly linearly with the cutoff (about 22 impressions per item per day).
  Training folds f1–f4 cover 1–4 days of data and val25 covers 5. Test covers about 7 days (cutoff Jan 27), so for the tree model test sits **outside the training range** on these columns. GBDT does not extrapolate; it just uses the last bin.
- Dividing by elapsed days is **not** a universal fix:

| feature | KS f1 vs val25, raw | KS f1 vs val25, ÷ days |
|---|---|---|
| it_clk | 0.587 | 0.256 |
| us_nev | 0.579 | 0.177 |
| it_nclk_all | 0.245 | 0.498 (worse) |
| it_nusers | 0.248 | 0.512 (worse) |

  it_nclk_all and it_nusers include the history-click pool, which reaches back before Jan 20, so their effective window is not "days since Jan 20". Each count needs its *own* denominator, which is why idea 2 below uses the global total of the same count.
- Total gain share of the cumulative group in the 325 model is only about 3.2%. The top cumulative columns are it_nclk_warm 0.0027, it_nclk_all 0.0026, v2_ip_per_imp 0.0023 and it_imp 0.0021; the rate/decay group is about 2.8%. So the upside is bounded. The risk, though, is concentrated on test (7 days vs at most 5 seen in training).

**B. The data is stationary at the item and day level (train_rows / train_events, Jan 20–26).**
- Events per day are 97.6k–98.5k. The event click rate is 0.626–0.632 on every weekday, Mon–Sun, so there is **no weekday effect**. The 67%-Monday test mix should not matter by itself.
- Correlation of an item's daily impression share is 0.914 at lags of 1, 2, 3 and 5 days (flat). Popularity is static plus noise; there is **no trend to extrapolate**.
- Correlation of an item's daily CTR is 0.06 at lag 1 and 0.038 at lag 5, so daily item CTR is almost pure noise.
- Method-of-moments beta prior for item CTR: true spread sd ≈ 0.011 around 0.063, prior strength a+b ≈ 490–540 impressions (4–7 days of data). On 1 day the true spread is not detectable at all.
  **Current K_ITEM = 20** (v3/features.py L29) and K_SMOOTH = 20 (v2work/feats.py L8).
- User event-click propensity: true sd ≈ 0.11–0.12 around 0.629, prior strength a+b ≈ 14–18 events (4–7 days). **Current K_USER = 5** (v3/features.py L30).

**C. Single-feature AUCs on val25 / val26.**

| feature | val25 | val26 | AUC type |
|---|---|---|---|
| it_ctr (K=20) | 0.5291 | 0.5259 | within-event |
| item CTR recomputed with K=250–500 | 0.5302 | 0.5261 | within-event |
| it_ctr24 | 0.519 | 0.513 | within-event |
| it_trend24 | 0.491 | 0.493 | within-event (below 0.5: noise) |
| stored us_ctr | 0.6014 | 0.6098 | event-level |
| us_ctr recomputed from us_nclk_win/us_nev, K=5 | 0.6068 | 0.6126 | event-level |
| us_ctr recomputed, K=10–15 | 0.6071 | 0.6136 | event-level |

**D. Bug found: us_ctr is wrong for cold users.** v3/features.py L256 does `pl.col('us_ctr').fill_null(g)`, where g = P_rows click mean = 0.063, a row-level CTR. Users with no prior events (about 3.3% of events: 1392/35040 in f4, 1468/43964 in val25) get us_ctr = 0.063. Their actual event click rate is 0.634 (f4) and 0.639 (val25), against 0.627 / 0.633 for warm users. The right fill is the event-level prior of about 0.629. The same line is in v2work/v3feat_ne.py L261, which builds the compliant submission_v3 features.

## 1. Ranked ideas (expected gain per hour)

Gains are "unquantified" unless measured above. None is a full-model result.

### 1. Fix the us_ctr cold fill and set K_USER to about 12 (0.5 h)
- **Expected gain:** single-feature event AUC goes up by 0.005–0.006 on val25 and 0.004 on val26 (measured, §0C/D). In the full model it is unquantified and probably small: us_nev = 0 lets trees isolate cold users, and us_ctr is constant within an event, so it only moves the global (cross-event) part of AUC. us_ctr is still the top non-embedding feature (gain 0.0063). The cost is near zero.
- **Sketch (in a copy, not in v3/):** `pl.col('us_ctr').fill_null(0.629)` and `K_USER = 12`. Do the same check for v2 `u_ctr` (feats.py L77 uses `5*g*10` with fill `g*10`, which is already event-level, so it is fine).
- **Rule risk:** none. Train events before C plus the user's own events.

### 2. Share-of-global normalisation for every cumulative count (1.5 h)
- **Mechanism:** replace X with X / ΣX over all items (or users) computed on the **same** train-only table at the same C. Optionally add the global percentile rank of the item within that window. In OTTO, the 1st-place solution reported that item popularity "worked well when ranked" and was "calculated by multiple time windows". The cumulative group is about 3.2% of gain (§0A), and test lies outside the training range on these columns.
- **Expected gain:** unquantified. Local val will understate it, because val25 is only 1 day beyond f4 while test is 3+ days beyond. Measure with idea 3.
- **Sketch:**
  ```python
  # item counts: it = CLg/P_rows aggregate per ii at cutoff C (already built in features.py ~L118-129)
  tot = it.select(pl.col(['it_imp','it_clk','it_nclk_all','it_nusers','it_nclk_warm']).sum())
  it = it.with_columns([(pl.col(c) / tot[c][0] * 1e5).alias(c + '_sh') for c in tot.columns] +
                       [pl.col(c).rank('average').truediv(pl.len()).alias(c + '_gpct') for c in ('it_imp', 'it_clk')])
  # user counts: us_nev / mean(us_nev over active train users at C); v2_ip_per_imp -> ip_n_sh / i_imp_sh
  # time-since: h_since_first / (t_event - T0) with T0 = 2025-01-20, ui_since_imp / (C - T0)
  ```
  Then drop the raw cumulative columns and their `_dm` / `_rmax` variants. Keep `_z` / `_rk`, which are already scale-free within an event.
- **Rule risk:** low, as long as the denominators come from CLg / P_rows before C. Never use R (the target rows) and never use eval rows. The one trap is computing `_gpct` over the candidate rows instead of the item table.
- **Note:** this replaces the "velocity / trend" family in practice. §0B shows popularity is static, so a stable *level* estimate is all that is needed.

### 3. Extrapolation-gap validation for count changes (1 h, one model fit per variant)
- **Mechanism:** train on f1–f3 only (1–3 days of counts) and validate on val25/val26 (5–6 days). This mimics the 2–3 day out-of-range gap that test has. Compare raw vs `_sh` feature sets (idea 2) and the drop-raw-cumulative ablation.
  This is different from the stale-like folds, which vary event staleness, not cutoff length.
  Precedent: the TalkingData 13th-place write-up warns that windowed click features must have the same maximum horizon in train and test, or there is "a clear difference between training and test features".
- **Expected gain:** none directly; it is the measurement that makes idea 2 decidable.
- **Rule risk:** none.

### 4. Drop or replace the noisy trend ratios (0.25 h, fold into the v5 null-importance pruning)
- it_trend24, it_trend6, it_imptrend24 and it_ctr24 / it_ctr6 compare a 24h/6h window against the total. With about 22 impressions per day and a true CTR spread of 0.011, the 24h CTR is noise (daily CTR autocorrelation 0.06; it_trend24 within-event AUC 0.49).
- **Expected gain:** unquantified, small (a variance reduction).
- **Sketch:** remove them from featlist and re-fit.
- **Rule risk:** none.

### 5. Empirical-Bayes item CTR with a fitted prior (0.5 h)
- **Mechanism:** beta-binomial posterior mean, with α and β fitted by method of moments per cutoff, following Micci-Barreca (2001) and the YunaQiu IJCAI-18 practice ("贝叶斯平滑 … 矩估计法": Bayesian smoothing with method-of-moments estimates). Measured prior strength is about 500 vs K = 20 now. Also give the posterior sd, a reliability term.
- **Expected gain:** single-feature within-event AUC +0.001 (0.5291 → 0.5302 on val25; +0.0002 on val26), so it is near-negligible on its own. Its real value is consistency across folds: the posterior mean has the same meaning at 1, 4 or 7 days.
- **Sketch:**
  ```python
  p = clk / imp; m = clk.sum() / imp.sum(); vt = max(p.var() - (m * (1 - m) / imp).mean(), 1e-6); ab = m * (1 - m) / vt - 1
  ctr_eb = (clk + m * ab) / (imp + ab)
  ```
  If vt ≈ 0 (as on 1-day folds), fall back to ab = 500.
- **Rule risk:** none (train-only).

### 6. Own-history timestamp shape features (1.5 h)
- **Mechanism:** IJCAI-18 top solutions used time since first/last/previous click and click counts in 10/15/30/45-minute and hourly windows (loyalzc and luoda888 repos).
  Here we would use only the row's own ≤20-click history: clicks within 1h/6h/24h before the event, last gap divided by the user's median gap (bursting vs dormant), gap coefficient of variation, and the number of clicks in the current "session" (gap < 30 min).
  The history is capped at 20, so these are bounded and stationary.
- **What already exists:** h_since_first, h_since_last, h_gap_mean, s_since_clk_h, s_clk_last1/3, s_nev24h. What is new is the *ratios* and the session/burst counts.
- **Expected gain:** unquantified. These are event-level (constant across the 10 candidates), so they mainly help the event-has-click part of AUC, where s_evrate 0.593 and us_ctr 0.601 are the current best.
- **Sketch:**
  ```python
  R.with_columns(gaps = pl.col('h_ts').list.diff())
   .with_columns(last_gap_ratio = (pl.col('ts') - pl.col('h_ts').list.last()) / pl.col('gaps').list.median(),
                 n1h = pl.col('h_ts').list.eval(pl.element() > pl.col('ts') - 3600).list.sum())
  ```
  In polars, pass the event ts in via explode + group_by, or via list.eval with a struct.
- **Rule risk:** none (own history only).

### 7. Recency-weighted candidate × own-history interactions (2 h, lower priority)
- Weight the history cosine by exp(−age/τ), where age = event ts − click ts from own history, with τ ∈ {6h, 1d}. Today hcos uses a position weight 1/√pos, and conr decays relative to C, not to the event.
- **Expected gain:** unquantified. Caution: x_conr1_k64 (decay relative to C, α = 1) already scores *lower* single-feature than con05 (0.6726 vs 0.7257), which is weak evidence against recency weighting here.
- **Rule risk:** none.

## 2. Researched but deprioritised (evidence says ≈ 0 here)
- **Item velocity / acceleration (24h vs previous 24h, slopes) and trend extrapolation to the event time.** Popularity autocorrelation is flat at 0.914 for every lag (§0B), so there is nothing to extrapolate. The TalkingData 13th place also found "previous day counts and ratios … not particularly effective".
- **Forecast-horizon features ("as of C−N", horizon as a feature).** Beyond the existing stale features, the per-day stationarity means features as of C−N carry the same information with less data. The stale-like folds were already flat.
- **Weekday / Monday handling.** There is no weekday effect in the event click rate (0.626–0.632 across all 7 days).
- **Time-aware out-of-fold / expanding target encoding.** Every TE-like stat (it_ctr, us_ctr, i_ctr_tb, cluster CTR) is already computed strictly before C for each fold. That is the time-ordered analogue of CatBoost ordered TS (Prokhorenkova et al. 2018), so there is no target leakage to remove; only the smoothing strength is wrong (ideas 1 and 5).
- **Counts from the event's own 10 candidates.** These are already covered by `_z` / `_rk` / `_dm` / `_rmax` and the ev_* aggregates. Anything counted across *other* events' candidates in eval is banned.

## 3. Sources (all opened / returned by search on 2026-10-23)
- TalkingData 1st place (next-click, future counts in 1h/6h windows, past attributed ratio): https://www.kaggle.com/competitions/talkingdata-adtracking-fraud-detection/writeups/flowlight-komaki-shuffle-1st-place-solution ; code: https://github.com/flowlight0/talkingdata-adtracking-fraud-detection
- TalkingData 13th place (bound windows so train and test have the same horizon; previous-day counts and ratios not effective): https://www.kaggle.com/competitions/talkingdata-adtracking-fraud-detection/writeups/alberto-danese-hints-on-13th-place-solution-and-fi
- OTTO 1st place (popularity over multiple windows, "worked well when ranked"): https://www.kaggle.com/competitions/otto-recommender-system/writeups/mrkmakr-1st-place-solution
- OTTO solution with normalised daily, same-day and weekly item counts: https://github.com/Fedorov-Artem/kaggle_OTTO/blob/main/README.md
- Outbrain 2nd place (counts conditioned on time: past vs future; competing ads per display): https://medium.com/kaggle-blog/outbrain-click-prediction-competition-winners-interview-2nd-place-team-brain-afk-darragh-610d072fa3fc
- IJCAI-18 feature lists (time-window click counts, first/last/prev/next click deltas, conversion rates): https://github.com/loyalzc/ijcai_2018 , https://github.com/luoda888/2018-IJCAI-top3
- IJCAI-18 Bayesian smoothing by method of moments for rate features: https://github.com/YunaQiu/IJCAI-18alimama
- Micci-Barreca 2001, empirical-Bayes target statistics: https://doi.org/10.1145/507533.507538
- CatBoost ordered target statistics: https://arxiv.org/abs/1706.09516
