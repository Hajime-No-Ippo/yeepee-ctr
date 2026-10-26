# v2work v6 leakage audit (parallel check, same method as parallel/report.md)
23 Oct 2026 (Europe/Dublin). Nothing in /workspace/huawei/v2work was modified, deleted or killed. No v2work jobs were running during the audit.
Everything here is in /workspace/huawei/parallel_v6/. Nothing was uploaded, submitted, committed or pushed.

## Verdict: **CLEARED WITH CAVEATS** (caveats are about selection bias and reproducibility, not leakage)
- **Confirmed local AUC: 0.80185 on val25** (cutoff Jan25; first event per user at or after Jan25; 439,640 rows). **0.81192 on val26** (cutoff Jan26; 351,380 rows).
- No label leakage found. No item, global or fitted stat uses eval rows (stricter than our own v3). The earlier minor flag (extra.py target-user vectors in iC/KMeans) is fixed in v6.
- Lag-1d test: 0.80116 → 0.79412 (−0.0070). The drop is gradual with no cliff, and smaller than v3-ne's −0.0128.
- **0.8019 is safe to show next to the 0.8046 public score.** Label it "local val, cutoff 25 Jan" (see caveats).

## Provenance (what produced submission_v6.csv)
- submission_v6.csv: sha256 0494968e…2b23, 529,390 unique rowids.
  - Equals v2work/eval_pred_v6.parquet and my recomputation from ens/v6{base,slow}_c27 (max abs diff 5e-9; subcheck.json).
  - Spearman vs submission_v3.csv = 0.906.
- Run: v2work/v6_run.sh (in solution_v6_src.zip, log v6_run.log 18:49–19:19).
  - extra.py → feat_xf (train-only Ug fix).
  - `ens.py {25,26,27} v6base --seeds 0,1 --lr feat_lr2`.
  - `ens.py … v6slow --seeds 0 --rounds 500`.
  - Blend: blend_v6.py.
- Code audited:
  - Embedded SRC of solution_v6.py, extracted to parallel_v6/emb/.
  - The zip copies in parallel_v6/src/.
- Diffs:
  - emb vs zip differ only by path placeholders and a cl.py cache-key/mods tweak.
  - vs the v3-audited snapshot: v3feat_ne.py, feats.py, prep.py, write_sub.py and validate.py are **identical**. v3fold.py, extra2.py and build_folds.py only add d-fold/lite specs. extra.py has the Ug fix.
  - New code: lrfeat2.py.
- Tables used:
  - feat_v3/feat/feat_x2 are from 04:06–04:27, the same tables cleared in the v3 audit.
  - feat_xf was built 18:49–18:51.
  - feat_lr2 was built 17:53–17:55 by lrfeat2.py (17:53, LR_KS 4..16, LR_CON=1; 54 lr cols).
- Stale-cache check (cachecheck.json):
  - All 9 cl.matrix caches used (f1..f6, val25, val26, eval) were written 18:51–19:08, after feat_xf was built.
  - Sampled cached cols equal the on-disk feat_xf/feat_lr2 values exactly.
  - So the run did not pick up a stale matrix. Note that cl.py at run time (L30 of src/cl.py) keyed the cache only on feat_v3's mtime, which is fragile. It is fixed in the embedded version (emb/cl.py L30).
- 379 features: 325 from v3-ne, plus feat_xf deltas, plus 54 lr*. The `ev_*` columns are within-event (same 10 candidates) rank/z/max/mean (emb/v3feat_ne.py L270–272), not eval aggregates. No rowid/eid, no mods (variant 'notime'; emb/cl.py L17–18).

## 1. Label leakage: NONE FOUND
- **Training targets before C.** ens.py L13 trains on f1..f(CUT−21): val25 → f1..f4 (targets Jan21–24), val26 → f1..f5, final → f1..f6.
  - v3/split.py L19–24: first_events = first event per user in [C, C+W).
- **v3/v2/x2 features:** unchanged from the cleared v3 audit (P_ev/P_rows filtered ts<C; asserts in extra/extra2).
- **feat_xf (emb/extra.py):**
  - L20 `P_ev = TE.filter(ts < C)`.
  - L22–24 CL/CLg plus `assert CL.ts.max()<C and P_rows.ts.max()<C`.
  - Cluster CTR (L59–67) uses P_rows labels before C only.
- **feat_lr / feat_lr2 (emb/lrfeat2.py):**
  - L19 SVD matrix = histories of *train* events with ts<C (TE only).
  - L24 `svds`.
  - L34–35 the "con" negatives = the user's own skipped impressions from train rows with ts<C.
  - The target row contributes only its given history list (L29–30) to its centroid.
  - z / dmax are within-event (L50–51).
  - The target row's own label is never read.
- **Own-history timing (checked on data):**
  - val25 target rows: 617,371 own-history entries, 0 with h_ts ≥ C.
  - eval: 708,389 entries, 0 with h_ts ≥ Jan27.
  - So "own history" never carries post-cutoff clicks.
- feat_tt / feat_xAn / seq / feat_rc / feat_ev / mods are **not used in v6**: extra=('feat_lr2',) only, and nf=379.
  - feat_xAn (our v5 contrast features) first appears in v7, not v6.

## 2. Eval-row usage: NONE beyond each row's own history (flag status: clean)
- Item/global/co-occurrence/SVD/idf stats come from CLg (train only): emb/v3feat_ne.py L114, L128, L173, L195; emb/extra.py L23, L31.
- **Fixed since the v3 audit:** item-side vectors iC/iN (emb/extra.py L52–53) and KMeans fit/labels (L59–60) now use train-only Ug.
  - The target row's own vector Un[uu] is only *assigned* to a cluster (L61). No pooling.
  - The effect is tiny: corr(old feat_x, new feat_xf x_icon05_k64) = 0.99999999.
- lrfeat2: SVD and negatives come from TE/TR (train) only. The eval build (L55) is fitted at Jan27 on train only and uses EE/ER only for each eval row's own history and candidates.
- Within-event z-scores/ranks over the 10 candidates of the same event are used. That is per-row input, not cross-row pooling.
- **Conclusion:** no eval-derived item or global stats or fitted transforms. Eric's open question about whether eval-row use is allowed does not affect v6.

## 3. Validation split and selection bias
- **Split:** val25 has 439,640 rows and val26 has 351,380. The rowid sets are identical to our v3/split and there are 0 label mismatches (score_val.json). Fit is strictly before C.
- **Blend:** fixed equal-weight mean of logits (blend_v6.py L5–12). No weights are fitted on val. The component choice (base 400r + slow 500r) is inherited from v4_lite.
- **Sensitivity on the reported rows (score_val.json):**
  - Weight grid w∈{0,.25,.5,.75,1}: val25 0.80141–0.80185, val26 0.81162–0.81192.
  - Round grid: val25 0.80153–0.80185.
  - The chosen config is at the top of both grids. The worst-case optimistic bias from blend/round choice is **≤0.0003 (val25) / ≤0.0003 (val26)**. Realistically it is about 0.0001, since the equal weight and the v4_lite rounds were fixed beforehand.
- **Broader caveat:** the v6 feature set (lr K grid, con variant) was chosen on val25/val26 screens (screen.py, lr_q1/lr_q2 logs, 17:00–18:10). The reported AUC is on the same rows, so there is some adaptive-selection optimism that can't be quantified. There is no untouched holdout (val23 wasn't run for v6). Public 0.8046 > local 0.8019 suggests this is not material.

## 4. Recompute and lag-1d (parallel_v6/rescore.py, lag/lagtest.py; 3 threads, one job, box otherwise idle)
**Recomputed from saved v2work preds** (row order taken from the run's own cache):

| set | base s0 | base s1 | base mean | slow 500r | **blend** | ≤1d | 1–2d | event AUC |
|---|---|---|---|---|---|---|---|---|
| val25 | 0.80116 | 0.80107 | 0.80167 | 0.80141 | **0.80185** | 0.80593 | 0.78457 | 0.6675 |
| val26 | 0.81128 | 0.81133 | 0.81174 | 0.81162 | **0.81192** | 0.81192 | – | 0.6723 |

All per-component values match v6_run.log exactly.

**Retrain plus lag-1d** (v6base config, seed 42, 400r, f1..f4 from the run's caches):

| val25 features | AUC | ≤1d | 1–2d | event AUC |
|---|---|---|---|---|
| as used in v6 run (C=Jan25) | 0.80116 (same as log s0) | 0.8052 | 0.78408 | 0.66605 |
| xf+lr2 rebuilt here from shipped code (C=Jan25) | 0.80110 | 0.80515 | 0.78400 | 0.66583 |
| **lag-1d: all stats frozen at Jan24, same targets** | **0.79412** | 0.79806 | 0.77746 | 0.66101 |

- The lag drop is −0.0070: gradual, spread across both staleness buckets, with no collapse. It is smaller than the v3-ne drop (−0.0128).
- Lag tables:
  - feat_v3/feat/feat_x2 reuse parallel/lag/ (same code as v6).
  - feat_xf and feat_lr2 were rebuilt with emb/extra.py and emb/lrfeat2.py, with only output paths and the lag spec patched (lag/extra_lag.py, lag/lrfeat2_lag.py).
- **Rebuild differences:**
  - lr2 cols corr 1.0000.
  - x_con/x_icon are effectively identical.
  - Cluster features differ because MiniBatchKMeans depends on thread count: x_clctr256 corr 0.77; x_climp* counts differ by up to 120. It doesn't affect AUC (0.80110 vs 0.80116). This is a reproducibility caveat only.

## Caveats
1. Adaptive selection on val25/val26 (see §3). Fixed blend/round choice bias is ≤0.0003.
2. val25 targets are ≤2 days stale, while eval spans Jan27–Feb3 (up to 7 days). The local number isn't a forecast of public. Empirically public has been higher (0.8046 vs 0.8019).
3. KMeans cluster-CTR features aren't bit-reproducible across thread counts (no AUC impact).
4. The run-time cl.py cache key was fragile. I verified the run used fresh caches, and it's fixed in the shipped solution_v6.py.
5. Per v2work/out/submission_log.csv, submission_v6.csv has val 0.80185 logged. The 0.804626 public score came from Eric and isn't in the log yet. This audit did not upload anything.

## Files
rescore.py, score_val.json, cachecheck.py/.json, subcheck.py/.json, lag/{extra_lag.py, lrfeat2_lag.py, lagtest.py, lagtest.json, lagtest.log, feat_xf/, feat_lr2/, model_v6base_f1to4_s0.txt, pred_*.parquet}, emb/ (extracted shipped sources), src/ (zip contents).
