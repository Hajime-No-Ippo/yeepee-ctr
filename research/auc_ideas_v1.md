# Yeepee: AUC improvement ideas v1 (Huawei TechArena 2026 Phase 1 CTR)
Written 2026-10-23 03:46 IST (Europe/Dublin). Research only: nothing in `v3/` or `v2work/` was modified.
The only new files are this one and `research/diag_decomp.py`, a read-only diagnostic that re-predicts val25/val26 with the saved `v3/out/full25.txt` and `full26.txt` models.
I fetched every link below with curl or WebFetch on 2026-10-23. ACM DL / doi.org / Medium return 403 to scripts, so I checked those through web search instead.
A dead link (`github.com/layer6ai-labs/RecSys2019`, 404) was dropped.
**"Unquantified"** means that no source gives a number that carries over to this data. Numbers marked *local* were measured on this box.

## 0. Diagnostic run for this note (local, read-only)
Saved v3 models were re-scored on their validation sets. Global AUC was split into an event-level part and a within-event part:

| | global AUC | event "has a click" AUC (sum of p) | within-clicked-event AUC (clicked item's rank) | top-1 acc | global AUC if event label were known (oracle) |
|---|---|---|---|---|---|
| full25 (val25) | 0.7556 | 0.649 | 0.745 | 0.322 | 0.851 |
| full26 (val26) | 0.7741 | 0.656 | 0.763 | 0.339 | 0.863 |

What this means: both parts leave headroom. The event-level part (37% of events have no click) is weak at about 0.65. Even with perfect event knowledge the score only reaches about 0.85, so within-event ranking has to improve too. The leaderboard gap (0.82–0.83) needs both.

Staleness coverage (local): the share of target rows with `tm_stale > 1 day` is **0% in f1..f5 and val26, 19.9% in val25 (max 2 d), and 33.4% in eval (max 7 d)**.

## 1. Gaps observed in the current v3 pipeline (from reading the code and logs)
1. **The training-fold population doesn't match eval.** `build_feats.py` makes folds f1..f6 as the *first event per user in [C, C+1d)*. Eval is the first event per user in [C, end), and a third of its rows are more than 1 day after C. Those are less active, longer-gap users. The model never trains on `tm_stale > 1d` or on those users, so trees extrapolate a constant for 33% of eval.
2. **Count features are not normalised for history length.** `it_imp`, `it_clk`, `it_nclk_all`, `us_nev`, `us_nclk_win`, `hist_imp` and others grow with `C − Jan20`. f1 has 1 day of fit window and eval has 7, so the same raw value means different things in each fold. Only `it_imp_rate` is normalised.
3. **Early stopping and reported AUC use the same validation set** (`train.fit`: `valid_sets=[dva]` with ES, then AUC on the same rows). That inflates the reported numbers a little. Use ES on val25 and report val26, or the reverse, or use a fixed number of rounds.
4. **The `lag1d` stress test (0.7391) probably overstates eval decay.** `sv_mode='lag1d'` drops the target user's *own* last-day clicks and negatives from the user centroids (`CL.filter(ts < C-DAY)`). In eval, the user's own clicks and impressions before C are all known, because they have no events between C and their eval event. Only *other users'* post-C activity and item-side statistics are missing. A fairer stress test freezes the item factors and popularity at C−Δ but folds in the user with all of their data before C (see idea 7).
5. **Leakage hygiene looks right.** The asserts in `leakcheck.log` are all zero. `notgt` (0.7570) ≥ `std` (0.7556), so the target's own history in the SVD doesn't leak. Every label statistic uses `P_rows` (ts < C). The leakcheck crash in the first `lag1d` run was a `None < int` assert on an empty frame, since fixed with `or 0`.
6. **The `mf` group adds nothing** (0.74897 → 0.74897). The logistic MF is only 6 Adagrad epochs, dim 24, trained on fit-window impressions, so it is undertrained or redundant. Replace it with a proper factor model (ideas 1 and 4) rather than tuning it.
7. **Co-occurrence and transitions are unweighted.** `cooc_tables` uses binary user–item co-occurrence with no time decay, no distance-in-sequence weighting and no session split. Transitions use only lags 1–3 with weight 1/lag. Otto-style weighting is missing (idea 6).
8. **Little repeat-click modelling.** In-history candidates are 2.6% of rows at CTR 0.30, which is about **12% of all positives**, and segment AUC is only 0.656. Present now: counts, last-click recency and position. Missing: inter-click intervals, decayed counts, user- and item-level *re-click rates* (idea 5).
9. **No event-level propensity model.** Event-click AUC is about 0.65 (table above). User CTR and history length exist, but there are no features about the user's *current* state (gap since last event, recent event click streak, events per day), and no explicit P(any click) model (idea 3).
10. **Training data volume:** only ~35k events per fold are used (first event per user per day), about 1.75M rows for cutoff 25, out of 6.87M available rows.

## 2. Ranked ideas (best expected gain per effort first)

### 1. Supervised per-user "taste vector" fold-in on fixed item embeddings (generalises `con05`)
- **What:** Fix item vectors V (SVD k64 now, iALS or item2vec later). For each user, solve ridge or weighted least squares on their impressions before C: `w_u = (V_Iᵀ W V_I + λI)⁻¹ V_Iᵀ W y`, where I is every item the user was shown (clicked = 1, skipped = 0 or −α) plus their click log (history and warm-up, positive). Features: `w_u·v_cand`, the same with `λ ∈ {1, 10}`, and the within-event z-score. Optional mirror on the item side: centroid of users who clicked the candidate minus centroid of users who skipped it, cosine with the user vector.
- **Why here:** the single most important feature, `ev_sv_k64_con05_z`, is a crude version of this (click centroid − 0.5 × skip centroid). A least-squares solve uses the full covariance and the 9 skipped competitors per event. That is exactly the user step of implicit ALS (Hu–Koren–Volinsky), with confidence weights.
- **Effort:** 2–3 h. **CPU:** about 54k small 64×64 solves per cutoff, under 1 min (batched `np.linalg.solve`). Final-fit budget is fine.
- **Leakage:** use only `P_rows` (ts < C) for y, as `con05` already does. Never include target events.
- **Validation test:** add the features, run `train.py --cutoff 25` and `--cutoff 26`, and check single-feature AUC against `sv_k64_con05` (0.709 single on val25).
- **Expected gain:** unquantified. Reasoning: `sv`, the group this extends, was +0.14, and its contrast feature carries 24–32% of gain.
- **Source:** iALS user-step formulation and tuning, [Rendle et al. 2021, "Revisiting the Performance of iALS"](https://arxiv.org/abs/2110.14037). Library: [benfred/implicit](https://github.com/benfred/implicit).

### 2. Make the training folds look like eval: first event per user until the end of train, plus denser cutoffs, normalised counts
- **What:**
  - (a) Rebuild folds with `W=None`, i.e. targets are the first event per user in [C, Jan27). Staleness then spans 0–6 days, like eval.
  - (b) Add half-day cutoffs (Jan21 00:00, 12:00, …) to get more distinct training events.
  - (c) Divide window counts by `ndays`, or add `/ndays` versions.
  - (d) Optionally drop f1 (1 day of history) or down-weight it.
- **Why here:** 33% of eval rows have staleness over 1 day, and the folds currently contain none. Long-gap users are also the less active ones, which shifts both the event-click base rate and feature distributions.
- **Effort:** 1–2 h, mostly rebuild time. **CPU:** about 40–55 s per fold (from `build_feats3.log`). Doubling to ~12 folds adds about 10 min to the full build. The final fit stays under 15 min if row-sampled.
- **Leakage:** none new. A target is always a user's first event at or after C, so its history is all before C. The same event can sit in several folds with different as-of features, which is fine. Group-split by event for any OOF.
- **Validation test:** retrain at cutoff 25 and 26. Report AUC by staleness bucket (≤1 d, 1–2 d) and a gap-reweighted AUC (idea 7).
- **Expected gain:** unquantified. It is a distribution-match fix, and it matters most on the private set, where a third of rows are stale.
- **Source:** H&M 1st place built train/valid as rolling weekly windows matching the test horizon and reported stable CV/LB. [H&M 1st place writeup](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/writeups/senkin13-30crmnsia-1st-place-solution)

### 3. Two-stage: P(any click in event) × P(item | click), used as features (not as a per-event normalisation)
- **What:**
  - (i) An **event model** (LightGBM, one row per event) predicts whether the event has a click. Inputs:
    - user state: `us_ctr`, recent-event click streak (clicks in the last 1/3/5 events), time since last event and last click, events in the last 6/24 h, history length, warm-up share, time bucket, day type
    - candidate-set summaries: max, mean and std of `con05`, `hcos`, `hist_in` count, item CTRs
  - (ii) A **choice model** trained only on clicked events (multiclass softmax over 10, or lambdarank/xendcg) predicts within-event share.
  - (iii) Feed `p_event`, `p_share` and `p_event × p_share` as features to the main binary LGBM, or use the product directly as a blend component.
  - Per-event softmax on *final* scores stays off-limits. Here the event-level multiplier carries the cross-event information.
- **Why here:** see §0. Event AUC is about 0.65. Within-event ranking alone gives about 0.85 with a perfect event signal. The two parts are learned better separately than through one binary loss with 63% no-click noise.
- **Effort:** 3 h. **CPU:** the event model is about 0.2M rows, under 1 min. The choice model is about 1.1M rows, about 1–2 min.
- **Leakage:** event labels for training come only from the target events of each fold. Recent-event streak features use only events before C. To feed stage-1 outputs into stage 2 inside the same fold, use OOF by event or user groups.
- **Validation test:** event AUC of `p_event` vs the current 0.649. Global AUC with and without the three new features.
- **Expected gain:** unquantified for AUC. The RecSys 2015 winner (buy or not, then which items) used exactly this two-stage decomposition as a key point.
- **Source:** [Romov & Sokolov, RecSys Challenge 2015 winner (doi)](https://doi.org/10.1145/2813448.2813510) and [code](https://github.com/romovpa/ydf-recsys2015-challenge)

### 4. Stronger and more diverse item embeddings: iALS, item2vec (SGNS on click sequences), with candidate-vs-history aggregates
- **What:**
  - **iALS** (`implicit`, factors 64–128, confidence `1 + α·count`, 15 iterations) on the as-of-C click log, plus the fold-in from idea 1.
  - **item2vec** (gensim Word2Vec skip-gram) on each user's time-ordered click sequence (warm-up plus window, before C). Tune window (3–10), `ns_exponent` (−0.5 to 1), epochs and sample.
  - **Features per embedding:** cosine to the last 1/3/5 items, max/mean over the last 20, time-decayed mean `Σ exp(−Δt/τ)` with τ ∈ {6 h, 2 d}, and the within-event z-score.
  - Optional: **EASE** item-item. A 50k × 50k dense matrix is about 10 GB in float32, too big for 15 GB RAM, so restrict it to the top ~15k items or skip it.
- **Why here:** latent similarity is clearly the dominant signal (SVD gave +0.14). v2work's extra SVD variants (`v2_svd24_cos_dm`) still added gain in the combined model (0.7617 vs 0.7556, *local*). Different factorisations make different errors. item2vec adds **order and locality** inside sequences, which a user × item SVD ignores.
- **Effort:** 3–4 h. `pip install implicit gensim`; neither is installed in `/workspace/venv`. **CPU (estimates):** iALS on about 54k × 50k with about 6–7M non-zeros takes well under a minute per cutoff on 8 threads. item2vec on about 11.7M tokens for 5 epochs takes a few minutes per cutoff. Train **once per cutoff** (about 7–12 cutoffs).
- **Leakage:** train only on clicks before C (the `CL` table). Do not train one embedding at Jan27 and reuse it for earlier folds.
- **Validation test:** single-feature AUC first (target ≥ `sv_k64_hcos` 0.694), then add to the full model at both cutoffs.
- **Expected gain:** unquantified here. Word2vec-for-recsys hyperparameters (negative-sampling distribution, epochs, subsampling, window) "can increase [performance] by an order of magnitude" on the paper's tasks, so the defaults should not be used. Tuned iALS was competitive with the state of the art on the benchmarks studied.
- **Sources:** [Item2Vec (Barkan & Koenigstein)](https://arxiv.org/abs/1603.04259); [Word2Vec applied to Recommendation: Hyperparameters Matter](https://arxiv.org/abs/1804.04212); [iALS revisited](https://arxiv.org/abs/2110.14037); [EASE (Steck)](https://arxiv.org/abs/1905.03375); [gensim Word2Vec docs](https://radimrehurek.com/gensim/models/word2vec.html)

### 5. Repeat-click model for in-history candidates
- **What:** for each (user, candidate) where the candidate is in the user's click log:
  - decayed count `Σ exp(−(ts−t_k)/τ)` with τ ∈ {6 h, 1 d, 7 d}
  - mean and min inter-click interval for that pair, and `(ts − last)/mean_interval`
  - number of impressions *since* the last click on it, and skipped-while-in-history count
  - **user re-click rate:** past impressions before C where the shown item was already in the user's log, and the fraction clicked (smoothed)
  - **item re-click rate:** the same, per item
  - rank of the candidate within the user's history by recency and by count
- **Why here:** in-history rows are about 12% of positives at CTR 0.30, but their AUC is only 0.656. `hist_n_all` is the #2 feature, so the model already relies on this signal coarsely.
- **Effort:** 2 h. **CPU:** joins on about 2.6% of rows plus one pass over `P_rows`, under 1 min per fold.
- **Leakage:** re-click rates use only impressions and labels before C. Intervals use only clicks before C, which the history guarantees for target events.
- **Validation test:** `segments.py` AUC on `cand_in_history` (0.656 now) and global.
- **Expected gain:** unquantified. Upper bound reasoning: this segment carries about 12% of positives.
- **Sources:** Otto feature list with per-item session recency, time-delta and count features ([Fedorov-Artem/kaggle_OTTO README](https://github.com/Fedorov-Artem/kaggle_OTTO)); repurchase as the strongest retrieval and feature source ([H&M 1st place](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/writeups/senkin13-30crmnsia-1st-place-solution)); repeat-aware modelling ([RepeatNet](https://arxiv.org/abs/1812.02646))

### 6. Otto-style weighted co-visitation and transition matrices (time-decay, distance weighting, direction, sessions)
- **What:** replace or augment `cooc_tables` with several matrices built from each user's ordered click log before C:
  - (a) Symmetric pairs within a window of ±k clicks, weighted `0.99995^|Δt_sec|`-style decay. Tune the base: a 1-day half-life is a sensible start, since sequences span weeks.
  - (b) Directional "next-click" pairs (i → j within 1–3 steps or under 6 h), weighted by `1/step`.
  - (c) Recency-of-pair weighting, so pairs closer to C count more (trend).
  - (d) Pairs inside "sessions" split at gaps over X hours.
  - Normalise by `pop_j^β` (β ≈ 0.5). Features: weight of the candidate from the last item, the last-3 sum, and the inverse-rank-weighted history sum (`Σ w(i_k→cand)/k`), plus the **rank** of the candidate in the last item's neighbour list.
- **Why here:** answers the "sequential and transition" request directly. The existing `co` group gave +0.011 before SVD. Ordered, decayed variants carry information that SVD cannot (direction and short-term drift).
- **Effort:** 2–3 h. **CPU:** a polars self-join over about 11.7M events with a window cap. Keep the top 50–200 neighbours per item, about 1–3 min per cutoff.
- **Leakage:** clicks before C only.
- **Validation test:** single-feature AUC vs `co_cos_max20` and `co_tr_last`, then the full model.
- **Expected gain:** unquantified here. In Otto these matrices were core candidate and feature sources for the top solutions. The 20th-place team tuned session gap, time base and rank weighting with optuna.
- **Sources:**
  - [OTTO 20th place (time-decay 0.99995^|Δt|, 900 s sessions, inverse-rank history weighting, type weighting)](https://www.kaggle.com/competitions/otto-recommender-system/writeups/mikhail-kamenshchikov-20th-place-solution-transfor)
  - [OTTO 1st place](https://www.kaggle.com/competitions/otto-recommender-system/writeups/mrkmakr-1st-place-solution) (several co-visitation matrices with different type and time weights, plus NN retrieval, then an LGBM ranker; summary via search, because the page body is JS-rendered)
  - [Deotte's candidate-rerank notebook](https://www.kaggle.com/code/cdeotte/candidate-rerank-model-lb-0-575) and [repo](https://github.com/cdeotte/Kaggle-OTTO-Comp)
  - [Fedorov OTTO README (click2click within 5 min, later-event weighting, exact-next matrix)](https://github.com/Fedorov-Artem/kaggle_OTTO)

### 7. Staleness-robust features and gap-weighted validation
- **What:**
  - (a) **Validation:**
    - Add a `val22` set: cutoff Jan22, first event per user until Jan27, train on folds before Jan22. It covers staleness of 0–5 days.
    - Report AUC per staleness bucket and an **eval-weighted AUC**: weight val rows so the bucket mix matches eval's `tm_stale` histogram (67% day 0), and do the same for user-gap buckets (eval median gap 31.5 h). Use `roc_auc_score(sample_weight=…)`.
    - Fix the `lag1d` test so it lags only item-side statistics and embeddings, and folds the user in with all of their data before C.
  - (b) **Features:**
    - Express time quantities relative to *event* time rather than C: `ts − last_click`, decayed counts evaluated at `ts`.
    - Add staleness-interaction-friendly features: `us_since_ev`, and `tm_stale` paired with item trend.
    - Use slower-decaying popularity (half-lives 1 d, 3 d, 7 d) next to the 6 h and 24 h ones.
- **Why here:** a third of eval is more than 1 day stale. The decay curve is unknown, and the current lag test conflates user-side and item-side loss. CTR is flat by day and time bucket (profile), so item drift may be small. That is worth measuring before investing more.
- **Effort:** 2 h. **CPU:** one extra validation build, about 5 min.
- **Leakage:** none new.
- **Validation test:** this item *is* the test. Then choose features by the weighted metric.
- **Expected gain:** unquantified. It mainly reduces public/private and local/LB mismatch risk.
- **Source:** H&M 1st place: time-windowed features (last week/month/season, time-weighted counts) and validation on the following week with stable CV/LB ([writeup](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/writeups/senkin13-30crmnsia-1st-place-solution)). The Dressipi 2022 overview notes that fast-changing preferences make history less representative ([Landia et al.](https://doi.org/10.1145/3523227.3552534)).

### 8. Within-event comparison features (candidate vs its 9 competitors)
- **What:** for the top 5–8 scores (`con05`, `hcos`, `imax20`, iALS, `hist_n_all`, `it_ctr`):
  - `x − event_max`, `x − second_best`, `is_event_argmax`
  - softmax share `exp(x/T)/Σ exp`, kept as a *feature*
  - the candidate's mean similarity to the other 9 candidates (event cohesion)
  - the count of competitors that are in the user's history
  - Keep the raw values as well, so cross-event level information survives.
- **Why here:** the `ev` group gave +0.006. Trees can't easily compute differences to the event max from separate columns. LogicAI's RecSys 2019 win relied on comparison features across the impression list.
- **Effort:** 1 h. **CPU:** negligible.
- **Leakage:** none (unlabelled, same-event).
- **Validation test:** ablate on cutoffs 25 and 26.
- **Expected gain:** unquantified. Probably small, around +0.002–0.005, but this is a guess and not sourced.
- **Source:** [LogicAI RecSys 2019 code (per-impression and comparison features)](https://github.com/logicai-io/recsys2019) and [paper doi](https://doi.org/10.1145/3359555.3359557)

### 9. Ensembling: seed/param bagging, a CatBoost/XGB member, feature-union over rank-blend; binary objective stays primary
- **What:** 3–5 LightGBM seeds with feature/bagging subsampling, plus one CatBoost model if installed (it isn't yet). Rank-average with weights tuned on val25 and checked on val26.
  - Prefer **feature union** over blending separately trained models. *Local* (v2work `blend_val.log`, val25): v3 alone about 0.7556, best rank-blend of v2 and v3 = 0.7584, combined feature set = **0.7617**.
  - Keep `binary`. *Local*: lambdarank 0.7485 vs binary 0.7556 at cutoff 25. Lambdarank only optimises within-event order and drops the event-level signal that global AUC rewards. A lambdarank/xendcg model is fine as a *stage-2 feature* (idea 3).
  - **submission_v1.csv:** it is a perfectly uniform rank transform with no matching validation predictions, so the blend weight can't be tuned and blending it is unvalidated. Blend only if its producing pipeline can be re-run at cutoffs 25 and 26.
- **Effort:** 1–2 h. **CPU:** 3–5× the fit time. With a 15-minute final budget, sample events or use lr 0.08.
- **Leakage:** tune weights on validation only.
- **Expected gain:** sourced reference only: H&M 1st place, single LightGBM private 0.0367 → 5 LGBM + 7 CatBoost 0.0371 (MAP@12, different metric). LogicAI blended 37 LGBM models. For AUC here: unquantified, typically small.
- **Sources:** [H&M 1st place](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/writeups/senkin13-30crmnsia-1st-place-solution); [LogicAI blend](https://github.com/logicai-io/recsys2019); [LightGBM objectives](https://lightgbm.readthedocs.io/en/latest/Parameters.html)

### 10. Small CPU sequential model (SASRec/gSASRec or GRU4Rec) as a feature generator, and the two-tower note
- **What:**
  - Next-click model on users' click sequences before C (max length 50, d = 64, 2 blocks, sampled softmax or gBCE).
  - **Initialise item embeddings from SVD or item2vec.** Feature: `seq_repr · emb(cand)` and its within-event z-score.
  - Train **per cutoff** (or at least at 25, 26 and 27, with OOF for earlier folds).
- **Why here:** captures order and recency jointly. It is the neural version of ideas 4 and 6.
- **Effort:** 6–10 h. **CPU (estimate):** tens of minutes per cutoff with torch on 8 threads. Too slow to run per fold inside the 15-minute final budget unless cached, so do it last.
- **Leakage:** train only on clicks before C for each fold. Never reuse a Jan27-trained model on earlier folds.
- **Expected gain:** unquantified. Reproducibility studies find simple kNN/graph or tuned linear baselines often match neural recommenders. BERT4Rec vs SASRec results are inconsistent, and gSASRec attributes SASRec's gap to overconfidence under negative sampling.
- **Sources:** [SASRec](https://arxiv.org/abs/1808.09781), [gSASRec](https://arxiv.org/abs/2308.07192), [BERT4Rec replicability](https://arxiv.org/abs/2207.07483), [GRU4Rec](https://arxiv.org/abs/1511.06939), [DIN](https://arxiv.org/abs/1706.06978), [DIEN](https://arxiv.org/abs/1809.03672), [Dacrema et al. 2019](https://arxiv.org/abs/1907.06902)
- **Why the two-tower (v2work `tt.py`) probably scored only about 0.556:**
  - Item tables (two of 50k × 64) were randomly initialised and trained only on fit-window impressions (events before Jan24).
  - That is about 137 impressions and about 9 clicks per item over the whole week, so there is far too little signal per parameter.
  - The user side saw only the last ≤20 history items. It had no user ID and no warm-up clicks beyond 20. The SVD, by contrast, uses the full 11.7M-entry click log including warm-up co-occurrence.
  - Training loss fell from 0.41 to 0.33 at epoch 3 while validation stayed at 0.556, which looks like memorisation.
  - NCF-style learned scorers are also known to underperform a well-tuned dot product ([Rendle et al. 2020](https://arxiv.org/abs/2005.09683)).
  - **Revive only as a thin learned layer on frozen SVD or iALS item vectors.** That means learning position and time-decay weights and a null-click head, which is effectively a learned `hcos`/`con`. Not worth reviving as a from-scratch ID model.

### (Low priority) Cold and empty-history users
- **Segments:** empty history is 1.3% of eval rows at AUC about 0.55. History-only users are about 2.7% of eval events at AUC 0.707–0.720.
- **What to add:** item CTR conditioned on `time_bucket` × `day_type` (smoothed toward the global item CTR), recent-trend popularity, and for history-only users the fold-in vector from idea 1, which works from history alone.
- **Effort:** under 1 h.
- **Expected gain:** tiny on global AUC, given the row share. Mostly check that these rows get a sensible *level* (CTR 0.045 vs 0.063) so they rank correctly across events.
- **Source:** [H&M 1st place](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/writeups/senkin13-30crmnsia-1st-place-solution) (recent-popularity fallbacks and cumulative features for inactive users).

## 3. Eval-row usage: ruled out by Eric (2026-10-23)
**Eric ruled out the "unlabelled eval rows" idea (disqualification risk).** The evaluation CSV may be used only to produce the final predictions. That means no aggregates over eval rows: no eval candidate frequencies, no SVD/ALS/item2vec/co-visitation fitted on eval rows or eval histories, and no popularity counts that include eval rows. Every idea above must fit its global and item-level statistics on training data only. An eval row's own history may still be used for that row's own user features at prediction time.

## 4. Top 3 to try first: implementation notes
**A. User taste fold-in (idea 1), in `features.build` after the SVD block**
```python
# V: (NI,k) item vectors (use V from k=64, *not* row-normalised and normalised variants both)
# per user u: items S_u = click log CL (y=1, weight c=1+a*count) ∪ P_rows impressions (clicked->1, skipped->0, weight 1)
A = V[items]; w = weights; y = labels
w_u = solve(A.T @ (w[:,None]*A) + lam*I, A.T @ (w*y))      # batch: group rows by u, use np.add.at to build per-user 64x64 Gram
feat 'fi_l{lam}_dot' = (W_u[uu] * V[iic]).sum(1); plus ev_ z-score; lam in {1,10}; a in {0, 2}
```
- Build the per-user Grams with `np.einsum` over sorted segments, or loop over users in chunks. 54k users × 64² floats is about 0.9 GB in float32, so chunk to roughly 5k users at a time.
- Cold or empty users get w = 0, so the feature is 0. Add an indicator.
- Assert `P_rows.ts.max() < C`.

**B. Eval-matched folds (idea 2), in a new `build_feats_v4.py` (copy, don't edit v3)**
- specs = `[(f'g{k}', J20 + k*DAY/2, None) for k in range(2, 2*(Ccut-J20)//DAY)]`. Half-day cutoffs, targets = first event per user until end of train.
- Add `x_per_day = x / ndays` for every count column.
- Train on g* with cutoff < val cutoff. Evaluate val25, val26 and a new val22 (`W=None`).
- Report the staleness-bucket AUC and the eval-weighted AUC:
  `w = eval_hist[bucket]/val_hist[bucket]`, then `roc_auc_score(y, p, sample_weight=w)`
- Pick the best iteration on val25 and report val26, so early stopping and reporting don't share a set.

**C. Event model + choice model (idea 3)**
- Event table: one row per target event. y = any click. Features:
  - `us_*`
  - streaks from `P_rows`: clicked flag of the user's last 1/3/5 events before C, time since last event, events in the last 24 h
  - `tm_*`
  - event max/mean/std of `con05`, `hcos` and fold-in
  - `hist_in` count
- Train LGBM binary on folds, giving `p_event` (OOF over folds for training rows). Choice model: LGBM `objective='multiclass'` is awkward, so use `lambdarank` on clicked events only, giving `s_choice`. Then `p_share = softmax(s_choice)` over the event, used as a feature only.
- Add `p_event`, `p_share` and `p_event*p_share` to the main binary model. Compare event AUC (baseline 0.649) and global AUC.
