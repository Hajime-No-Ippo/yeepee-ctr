# Yeepee v5 experiment ideas (2026-10-23, ~14:45 IST)
Base pipeline: v2work "ne" (feat_v3 via v3feat_ne.py + feat (v2) + feat_x (extra.py) + feat_x2 (extra2.py), combo.py notime, 325 features, LightGBM).

## What I read before ranking
All numbers come from existing files or quick reads on val25 at nice 19. No training was run.
- **Gain importances** (`eval_pred_v3_model_s0.txt`, 325 features):
  - Top features:
    - x_con05_k32_z 0.221, x_icon05_k64_z 0.085, ev_sv_k64_con05_z 0.083, x_icon05_k64 (raw) 0.046
    - v2_svd16_cos_dm 0.036, x_con05_k128 0.028, sv_k64_con1 0.022, r_l3n 0.022, x_con05_k32 (raw) 0.021
  - By group: x/z 0.349, v2/raw 0.149, ev/z 0.132, x/raw 0.108, sv/raw 0.095.
  - 11 features have zero gain and 42 have gain < 1e-4.
- **Single-feature AUC on val25** (global / within-event, i.e. rank of the click among its 10):

  | Feature | Global | Within-event |
  |---|---|---|
  | x_con05_k32 | 0.7184 | 0.7257 |
  | x_con05_k64 | 0.7134 | 0.7174 |
  | x_con05_k128 | 0.6896 | 0.6910 |
  | x_icon05_k64 | 0.7131 | 0.7141 |
  | x_icos_k64 | 0.6850 | 0.7107 |
  | x_conr1_k64 | 0.6705 | 0.6726 |
  | r_l3n | 0.6860 | 0.6957 |
  | x_con05_k32_z | 0.7260 | 0.7257 |

  - Pearson correlations: x_con05_k32 vs k64 0.73, vs k128 0.64, vs x_icon05_k64 0.65. The embedding settings matter and the families differ.
  - x_con05_k64 is identical to sv_k64_con05 (max |diff| = 0): a duplicate column.
- **Event-has-click AUC** (val25, 43,964 events, 63.35% have a click): max x_con05_k32 0.589, mean 0.584, std 0.554, s_evrate 0.593, us_ctr 0.601. The eventmodel.py full model reaches 0.649.
- **ens_log.jsonl (Grok Bot):**
  - val25 base seeds 0.7772–0.7778 @400; val26 base 0.7934–0.7938.
  - wide: no better. slow: +0.0005 (0.77815 @500).
  - nov2 (dropping the 108 v2 features): 0.77401 vs 0.77748 (seed 0), i.e. −0.0035. The second embedding/co-click family adds real value.
- val25 and val26 are all weekend rows (combo seg weekday n = 0). Eval starts Monday 2025-01-27.

## Already covered, so dropped
| Suggested direction | Why it's dropped |
|---|---|
| Raw (un-z-scored) copies of the top contrast features | They already exist (x_*, *_rk, v2 *_dm, *_rmax) |
| Event-level mean/max/std of contrast features | Already in eventmodel.py (x_*_emax, _estd); stacking added +0.0003 (screen 'ev') |
| Recency-weighted contrast | x_conr05/conr1/crcos_k64 exist; gain ≤ 0.002; conr1 single-feature within 0.673 < 0.717 |
| Skipped-impression (negative) centroid | Already inside con05 (`extra.py` Sn) and r_l3n (ridge, y_neg = −0.5) |
| Time-since-last-event / activity rate | Already present: s_since_ev_h, s_since_clk_h, s_evrate, s_nev24h, us_since_ev, v2_h_gap_mean |
| Deeper LightGBM, seed bagging | Grok Bot's ens.py owns this; wide was no better |
| Supervised MF (mf_*) | Exists; total gain 0.003 |
| Two-tower model | Weak (tt_exp: AUC 0.556) |
| Stale folds, rcov, lambdarank, choice/event stacking, per_day | Owned by Grok Bot or already failed in v4 |

## Ranked v5 experiments

### 1. Sweep the contrast embedding (one-feature screen first, then add the 2–4 best)
- **Hypothesis:** the top feature (con05, user click centroid − 0.5·skip centroid vs candidate) depends strongly on how the SVD is set up, and that has never been searched.
  - Dimension k: within-event 0.7257 / 0.7174 / 0.6910 for k = 32 / 64 / 128.
  - Correlations of only 0.64–0.73 between k values suggest a better setting and complementary settings both exist.
- **Change:** a new `v2work/con_sweep.py` that copies `extra.py` `build()` lines 20–38 and computes only on val25.
  - Grid:
    - k ∈ {16, 24, 32, 48}
    - weighting ∈ {tf-idf (current), binary without idf, BM25 (k1 = 1.2, b = 0.75), log1p(count)·idf}
    - singular-value exponent β ∈ {0, 0.5 (current `Vt.T*sqrt(S)`), 1}
    - negative weight α ∈ {0.25, 0.5, 1}
  - Score each by the global and within-event AUC of raw and `_z`, plus correlation with x_con05_k32.
  - Keep the 2–4 best that also have correlation < 0.9 with x_con05_k32. Add them to `extra.py` as `x_con{α}_k{k}_{w}` with `_z`/`_rk` columns, then rebuild f1..f6, val25, val26 and eval (`extra.py` takes 9–37 s per table; see `build_d.log`).
- **Global-AUC mechanism:** mainly within-event ranking. The raw columns also help calibration between events.
- **Expected gain:** unquantified. Single-feature gaps of 0.003–0.035 between existing variants, and the −0.0035 when the second family (v2) is dropped, show the model rewards better and more varied embeddings.
- **Cost:** the sweep is about 150 configurations × (one randomized SVD on CLg + fold-in) at 3 threads, roughly 15–30 min single-process. Reduce it by fixing α = 0.5 for the first pass (48 configurations). Then a rebuild (a few minutes) and one 3-seed combo run.
- **Leakage check:** SVD, idf and BM25 statistics come from CLg, which is train-only. Sn comes from P_rows, the train impressions before C. A user's own centroid uses CL (category (b), its own history). Use BM25 document lengths from CLg rows only.

### 2. Item-side contrast at the best k, and α variants
- **Hypothesis:** the item-side contrast is the #2 feature (gain 0.085 for z + 0.046 for raw; within-event 0.7141; correlation 0.65 with the user-side contrast). It is only computed at k = 64 with α = 0.5, while the user side is best at k = 32.
  - Item-side contrast = for item i, the centroid of users who clicked i minus 0.5 × the centroid of users who skipped i, compared with the target user's vector.
- **Change:** in `extra.py`, inside the k loop, add `x_icon05_k32`, `x_icon1_k32`, `x_icon025_k64`, `x_icos_k32` (+ `_z`/`_rk`). Optionally add a two-sided score: `nrm(iC − α·iN)[ii] · nrm(nC − α·nN)[uu]` after mapping both into the same Vn space.
  - Also build `Un` for iC/iN/KMeans from CLg (this fixes the code-level risk from the audit at `extra.py` L51/L55–58).
- **Global-AUC mechanism:** within-event ranking; the raw versions help between events.
- **Expected gain:** unquantified; it's the same kind of argument as #1. Do it in the same build as #1.
- **Cost:** marginal on top of #1 (two sparse matmuls per k).
- **Leakage check:** Scg and Sn are train-only. The target user's Un is a fold-in from its own history (b). No eval candidates are pooled.

### 3. Taste-confidence magnitudes (between-event calibration)
- **Hypothesis:** `nrm()` throws away how concentrated a user's taste is. A user with a sharp click centroid and many observations should get confidently high scores on matches. Global AUC needs exactly that cross-event level, and z-scores remove it.
- **Change:** in `extra.py`, for k32 (and the best configuration from #1), add:
  - `x_con05_k32_norm` = ‖nC − 0.5·nN‖ before normalising
  - `x_con05_k32_dot` = the un-normalised dot product
  - `x_nclk_cl`, `x_nskip_p` = the number of clicks and skips behind each centroid
  - `x_con05_k32_dmax` = raw − event max
  - Only `sv_k32_hnorm` exists today (gain 0.0023), and only for the history vector.
- **Global-AUC mechanism:** calibration between events.
- **Expected gain:** unquantified and probably small. For reference, the event max of x_con05_k32 alone gives event-has-click AUC 0.589, while the whole event model only reached 0.649 and added +0.0003 when stacked.
- **Cost:** negligible; same build.
- **Leakage check:** per-user computation from that user's own CL and train skips (b).

### 4. Slate-context features (the event's own 10 candidates)
- **Hypothesis:** synthetic slates are probably partly personalised and partly filler. A candidate close to the slate's centroid, or to the slate item that best matches the user, carries signal the per-candidate contrast misses. The match between the user and the slate centroid is an event-level propensity.
- **Change:** a new `v2work/slate.py` → `feat_sl/NAME.parquet`, using k32 Vn from CLg:
  - `sl_cand_cent_cos` = candidate vs mean of the other 9
  - `sl_disp` = slate mean pairwise cosine
  - `sl_user_cent` = con05 user vector vs slate centroid
  - `sl_cand_top_cos` = candidate vs the slate's top-con05 item
  - `sl_pop_rank` = item popularity rank within the slate (train-only popularity)
  - Pre-check first: the single-feature within-event AUC of `sl_cand_cent_cos` on val25 (one SVD, a few seconds). Drop the idea if it is ≤ 0.55.
- **Global-AUC mechanism:** both (within-event ranking via candidate centrality, between-event via user–slate fit).
- **Expected gain:** unquantified; this is a new direction, and the pre-check decides.
- **Cost:** about 1 minute per table.
- **Leakage check:** only the row's own event candidates are used (the same scope as the existing within-event z-scores). Item vectors and popularity are train-only. No statistics across eval events.

### 5. Sequence-window PPMI-SVD embeddings (item2vec equivalent, no new installs)
- **Hypothesis:** gensim is not installed, but SGNS is roughly equivalent to factorising shifted PMI (Levy & Goldberg, NeurIPS 2014, "Neural Word Embedding as Implicit Matrix Factorization"). Time-local co-click context is a different embedding family from user×item tf-idf SVD. The v2 family is evidence that a second family pays off (nov2 −0.0035).
- **Change:** a new `v2work/ppmi.py`:
  - From CLg sorted by (u, ts), count item pairs within a ±5-position window (weight 1/distance).
  - Compute shifted PPMI = max(PMI − log 5, 0), then randomized SVD with k = 32 and 64.
  - Features: `x_pp_con05_k32` (contrast as in #1), `x_pp_last3` (max cosine to the last 3 history items), `x_pp_hcos` (+ `_z`/`_rk`).
- **Global-AUC mechanism:** mostly within-event ranking.
- **Expected gain:** unquantified.
- **Overlap:** check with Grok Bot. This is not the directional co-visitation score in rcov or the DIN in seq.py, but it is related.
- **Cost:** item×item sparse counts plus an SVD, a few minutes at 3 threads.
- **Leakage check:** CLg only (train clicks and train-event histories before C). Never include eval histories in the window counts.

### 6. Remove duplicates and prune by null importance
- **Hypothesis:** 325 features include an exact duplicate (x_con05_k64 = sv_k64_con05), 11 zero-gain features and 42 with gain < 1e-4. Noise features slightly hurt GBDT generalisation.
- **Change:** in `combo.py`/`cl.featlist`, drop exact duplicates. Then train 1 model on real labels and 5 on shuffled labels (25% user sample via `screen.py` settings, 300 rounds). Drop features whose real gain is below the 90th percentile of their shuffled gain.
- **Global-AUC mechanism:** neither directly; it's variance reduction.
- **Expected gain:** about 0, possibly slightly positive. Risk: the screen 'dropcum' (fewer features) lost 0.0009 / 0.0005 on val25 / val26. Accept only if both cutoffs improve.
- **Cost:** 6 screen runs of about 1–3 min each, one at a time.
- **Leakage check:** none; labels come from the training folds only.

### 7. A different learner as a blend member (LightGBM DART; CatBoost needs pip install)
- **Hypothesis:** model diversity helps a blend even when single-model AUC is equal. Coordinate with Grok Bot, which owns ens.py.
- **Change:** `ens.py` with `--params '{"boosting":"dart","drop_rate":0.1,"skip_drop":0.5}' --rounds 600`, saved as an ens member. A CatBoost depth-8 run only if Eric approves installing it.
- **Global-AUC mechanism:** both, through averaging.
- **Expected gain:** unquantified. Evidence that capacity isn't the bottleneck: wide was no better and slow gave +0.0005.
- **Cost:** DART has no early stopping and runs about 2–3× slower than gbdt: tens of minutes at 3 threads. Do this last.
- **Leakage check:** the same features as the base model.

## Guard for every v5 candidate
val25 and val26 are both weekend cutoffs (Sat and Sun). Eval starts on a Monday. Also confirm each accepted change on a weekday cutoff (for example val22 or val23 with the d/e folds Grok Bot already built), or at least make sure it doesn't regress there.

Order: #1+#2+#3 in a single extra.py rebuild, then #4's pre-check, then #5, then #6, then #7.
