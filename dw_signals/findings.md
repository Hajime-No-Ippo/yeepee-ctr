# New-signal search for Huawei CTR (v11 baseline) — findings

Eric: 2026-10-27, Europe/Dublin (UTC+0). Everything is read-only against others' files; all outputs are in this directory.

## Bottom line
**Nothing clears +0.002 incremental AUC, alone or combined.** The best single family is **impression-space taste embeddings**
(SVD of who-was-shown-what, not who-clicked), worth about **+0.001** over v11. The whole new set together gives
**+0.0014 to +0.0015 at C25 and C23** (95% CI roughly [+0.0011, +0.0018]). That is real, since the decoy control reads ≈0, but small.
Expect about +0.001 on the public LB. It will not get 0.815 to 0.82 by itself.

## How incremental value was measured
- Cuts C=25 and C=23. Features come from train events with ts < cut plus each row's own pre-ts history. Validation rows are the first event per user in [C, Jan27), the same as v10/v11.
- v11 prediction recipe z=(sCbig[:,1]+sC_s[:,1]+2·mean(mlp2_s0..2[:,3]))/4 gives C25 .81524 and C23 .79424, which reproduces the team numbers.
- **Method 2, residual boosting (used for all reported deltas).** LightGBM with `init_score` = v11 logit, lr .03, 15 leaves, 200 rounds, 5-fold GroupKFold by user, 2 seeds, plus v11-derived context features (within-event z/rank, event lse/max/mean, hist_len, nohist). Δ = AUC(with candidate features) − AUC(same model without them). The paired event-cluster bootstrap CI uses 60 resamples. `delta_vs_v11` is measured against raw v11. The resid-base alone is +0.0001 over v11, so the baseline loses nothing.
- **Decoy control:** 5 existing v11-style features (sv_k64 ucos, mf score, item CTR, co-click cos max20, x_clctr64), recomputed independently. They give **+0.00003 (C25) and −0.00000 (C23)**, so the harness is unbiased.
- Method 1 (stack of v11 logit and the features, no init_score) at C25 gave the same ranking. It is biased by about −0.0008 because the stacker sees only about 350k rows, so it is not used. Results are in `stack_results.jsonl` (method=resid for the headline numbers).
- Test noise: the CI half-width is about ±0.0002 for the big groups. A 3.6k-row no-history segment is very noisy (±0.01), so ignore its signs.

## Ranked table
Standalone = global AUC over all validation rows (within-slate AUC in brackets). v11 itself is .8152/.7942. Δ = incremental over v11 (residual boosting).
"stale" = rows whose last history is ≥24h old (48h+ in the last column, where measured; C25 has too few 48h+ rows for a stable number).

| # | Signal (group) | Standalone C25 [C23] | Δ C25 (95% CI) | Δ C23 (95% CI) | Δ stale24+ C25 / C23 | Δ stale48+ C23 |
|---|---|---|---|---|---|---|
| 1 | **Impression-SVD taste family**: user-vec·item-vec cos at k=8/16/32/64/128, plus cos of own click-history centroid in impression-item space (k=8/32/64/128), plus max-sim to history items (k=64) (`imp23`) | .57–.60 | **+0.00116** [.00088,.00146] | **+0.00097** [.00070,.00121] | +.0011 / +.0011 | +.0009 |
| 1a | ...only the k=32 user·item cos (`impsvd`) | .578 [.568] | +0.00068 [.00047,.00089] | +0.00051 [.00030,.00067] | +.0008 / +.0008 | +.0003 |
| 1b | ...k=16/64 cos, skip-only SVD cos, hist centroid in imp space (`imp2`) | .565–.59 | +0.00102 [.00067,.00129] | +0.00064 [.00041,.00088] | +.0011 / +.0009 | +.0008 |
| 1c | ...k=8/128 cos, hist-centroid k=8/64/128, hist max-sim k=64 (`imp3`) | .57–.60 | +0.00065 [.00040,.00086] | +0.00069 [.00044,.00091] | +.0006 / +.0010 | +.0009 |
| 2 | **Slate centrality**: candidate's mean/max/top3 cos to the other 9 candidates, in click-SVD and in impression-SVD spaces, plus slate coherence (`central`) | .536–.554 | +0.00055 [.00030,.00078] | +0.00045 [.00024,.00063] | +.0009 / +.0010 | +.0010 |
| 3 | Skip (shown-not-clicked) centroid contrast in click space: c_skall, c_skrec, c_impall, c_hist, c_hist−c_sk (`skip`) | .55–.59 (c_hist .747 is already in v11) | +0.00018 [−.00000,.00036] | +0.00028 [.00017,.00044] | +.0008 / +.0005 | +.0008 |
| 4 | Joint click+impression SVD, click-part item vecs (`joint`) | .6425 [within .657] | +0.00008 [−.00004,.00022] | +0.00020 [.00007,.00035] | +.0002 / +.0004 | +.0005 |
| 5 | Exposure Pearson-residual / ratio, skip count/frac (`expo`) | .50–.527 | +0.00003 [−.00007,.00011] | +0.00018 [.00010,.00032] | +.0001 / +.0003 | +.0004 |
| — | **All new features together (ALL3, 61 cols)** | | **+0.00146** [.00111,.00177] | **+0.00143** [.00118,.00168] | **+.0016 / +.0019** | **+.0016** |
| — | v1 ALL (35 cols: skip, impsvd, central, expo, usertime, coimp) | | +0.00118 [.00093,.00143] | +0.00108 [.00078,.00127] | +.0015 / +.0015 | +.0011 |

Segments (ALL3, Δ): history-only/with-history rows **+0.0015 (C25) / +0.0014 (C23)**. The no-history slice is only 3.6k rows at C25 and is unstable: +0.0005 at C25, +0.019 at C23 (the C23 value is 3× noisier; treat as "maybe positive"). The 48h+ stale slice at C23 gains **+0.0016**, so the new signals help most where v11 is weakest (stale AUC .767), but only about twice the average.
Not run: the cold-user (no train rows) slice, and a separate 3-seed bootstrap. Everything used 2 seeds.

## Definitions (see `signals.py` / `reusable.py`)
1. **Impression SVD**: M[u,i] = number of times item i was shown to user u in train events with ts<X (clicked or not, all 10 slots). Scale by 1/sqrt((nu+1)(ni+1)), take `svds(k)`. Features: cos(U·s, V·s) for the row's user and candidate; centroid of the row's own click history (pre-ts) in the normalised item space, cos to candidate; max-sim over history items; within-slate z-scores. Slates are personalised (retrieval is conditioned on the user), so impressions carry taste information that clicks don't. The impression matrix has roughly 10× the click density, which helps low-history users. Why it adds over v11: v11's SVD/co-click features are built only from clicks.
2. **Slate centrality**: G = E·Eᵀ over the 10 candidate embeddings of the event (click-SVD 32 or impression-SVD 32). mean/max/top3 over the other 9, demeaned mean, and slate mean pairwise cos. Uses only the row's own slate, so it is pure within-event.
3. **Skip contrast**: centroid (click-SVD space) of items shown to the user and not clicked (all train, and last 3 days), cos with candidate; history-centroid cos minus skip-centroid cos.
4. **Joint SVD**: hstack([3·clicked, impressions]) user×item matrix, k=32, click-part item vectors.
5. **Exposure residual**: (u_imp·i_imp/N expected) vs observed user-item exposures, skip count and fraction.

## Leak-safety
- Every aggregate (impression matrix, SVD, click log, co-impression, exposure) comes from `D.snap(X)`/events with ts < X (the fit cutoff). No validation or eval rows (clicks, impressions, histories) enter any aggregate. For eval, X would be the end of train, and the eval row's own history is the only per-row input.
- Per-row inputs are the row's own user id, candidate, ts and own pre-ts history; centrality uses the row's own slate only (the 10 candidates are available at inference).
- User-level aggregates over train events are keyed by user id, the same exposure level as v11's own features. Check: the numbers hold on both C25 and C23 with the same code, and on the decoys.
- One caveat to verify before use at eval: the impression matrix needs the **train events' full 10-candidate slates** (they exist in `lite/train_events.parquet`). For a final fit use train through the end of train; staleness grows, so the ~0.001 gain should be checked on the C23-like (older fit) cut, which it was.

## Compute cost
- All base signals on one cut: about 25–35 s total on this box (snap 1–2 s, click SVD about 5 s, 5 impression SVDs about 3–8 s each, centrality seconds, plus the 16/64/128-rank ones). Memory < 6 GB. `test_reusable.py` reproduces the cached features exactly (42 cols of the recommended set, 0 mismatches).
- Incremental test: about 1 min per group per cut (2 seeds × 5 folds).

## Ruled out (Δ ≈ 0 or within noise of decoy)
| Signal | Δ C25 | Δ C23 | Note |
|---|---|---|---|
| Co-impression counts/lift over independence (history item × candidate in same train slate) | −0.00003 | +0.00004 | standalone .52 |
| User time features: hour-of-day match of clicks/events, last-event-clicked, recent-5 event click rate, same-bucket event CTR | −0.00003 | +0.00008 | ev_ctr_last5 has event-level AUC .585 but v11 already captures it |
| Recent (1d/3d) impression centroid + ui exposures in window | −0.00003 | +0.00001 | |
| Skip-only features on their own | +0.0002 | +0.0003 | small positive, not worth the cost alone |
| Exposure residuals / skip counts | +0.00003 | +0.00018 | |
| Joint click+impression SVD | +0.00008 | +0.0002 | |
Not tried: item-sequence signals (seq, PPMI, co-click, exposure-count features are already in v10/v11), per-user ranking calibration beyond what v11 does, train→eval drift beyond staleness (the stale mix is already in the table).

## Why no-history users stay near-random
No-history rows are 0.8% (C25) to 1.3% of validation rows. On them the best standalone features are .52–.55 AUC: impression-SVD (user has train impressions but no clicks) at k=64/128 (.55), cent_imp_top3 (.55), and ev_ctr_last5 (.556). Global popularity is already in v11. The new features lift them a little, but the segment is too small to move global AUC (+0.0001 at most). Zero-history rows with no train rows (cold, C23 ≈ 10%) have nothing but slate-level information (centrality, popularity).

## Recommendations (expected gain)
1. Add the **impression-SVD family** (`reusable.impression_taste`) as stacker/boosting inputs, especially k=64 hist max-sim and the cos at k=16/32/64. Expected **+0.0008 to +0.0012**.
2. Add **slate centrality** (clk + imp). It is cheap and almost uncorrelated with #1. Expected **+0.0004 to +0.0006**, mostly additive.
3. Add skip contrast only if it is free: **+0.0002**.
4. Together expect **+0.0012 to +0.0015** on top of v11. Ensemble through residual boosting on v11 logits (init_score) rather than raw stacking; raw stacking loses 0.0008 to the baseline.
5. To reach 0.82, other levers are needed (event-level "any click" calibration is AUC about .67 vs within-event .80; the stale slice). These features don't change the picture.

## Files
- `reusable.py` — clean entry points: `impression_taste(D,X,rows)`, `slate_centrality`, `skip_contrast`, `all_recommended` (fit cutoff X + rows table → feature DataFrame). `test_reusable.py` checks them against the cache.
- `signals.py` — all signal implementations; `common.py` — data loader, snapshot click log, SVD, helpers.
- `compute.py`, `compute2.py`, `compute3.py` — per-cut feature caches (`cache/sig_C*.parquet`, `sig2_`, `sig3_`, `extra_` decoys).
- `standalone.py`, `standalone2.py`, `cache/standalone_*.txt` — standalone AUCs.
- `stack.py`, `run_stack.py` (method 1), `run_stack2.py` (method 2), `groups.json`, `mk_decoy.py`, `stack_results.jsonl`, `show.py`, `cache/stack2*.log`.
