# v17cooc — TRANSDUCTIVE item co-occurrence (PPMI→SVD) features, teammate-p33 style  (variant B = PRIMARY, TRANSDUCTIVE)

**Variant B uses unlabelled evaluation-row aggregates** (`history_item_ids` / `history_timestamps` of the 52,939 evaluation events are added to the item co-occurrence graph). Normally banned by Eric; enabled by Eric ("Just add it") for this feature only. No evaluation *label* exists/is used. **Variant A** (train-only graph = DW's leak-safe build) is the comparison. **Nothing was uploaded to the leaderboard.**

## What is added (25 columns; feature names are `coA_*` for A and `coB_*` for B, selected from the 58 built by `mkcooc.py`)
Graph: item×item symmetric co-occurrence over each user's time-ordered, minute-deduplicated click log (window W=20 click positions, weight 1/d). PPMI (context smoothing α=1.0, shift 0) → randomized SVD (24 comps, n_iter 7, seed 0), whitened coordinates (p=0). User vector = centroid of the unit item vectors of the row's own pre-timestamp history. No recency weighting (uniform centroid).
- cosine group (13): `pp12_cos` (k=12, main), `pp12_cos_z`, `pp12_cos_dm` (within-event z / demeaned), `pp8_cos`, `pp8_cos_z`, `pp16_cos`, `pp16_cos_z`, `pp12x_cos` (cols 1..12), `pp12x_cos_z`, `ppb12_cos` (W=60 weight 1, un-whitened SVD→12), `ppb12_cos_z`, `pp12_unorm` (‖user centroid‖), `pp12_ucos_dot` (cos×norm)
- product group (12): `pp_prod0 … pp_prod11` (user_d × item_d, unit vectors, k=12)
(v8 already has PPMI k=32/64 features; those are not duplicated. Recency/position/user-coordinate/multi-config groups were built in `mkcooc.py` but are NOT used.)

## Variants
| | click log for the graph at cutoff X |
|---|---|
| A (train-only) | train events with ts<X: their histories + their clicked candidates (labels of ts<X only) — exactly DW's `D.snap(X)` build; reproduces `dw_p33/cache/pp_c25.parquet` bit-for-bit |
| **B (TRANSDUCTIVE)** | A's log ∪ all history entries (user,item,minute; deduplicated) of the events in the TARGET table (`eval`: 52,939 evaluation events; fold/val tables: that table's own events = exact mirror). Unseen eval users get a private pseudo-id per event (no fake merged user). No clicked label of any target-window event feeds the graph |

Fit cutoffs: f_k = J20+k·day; val23/25/26 = J20+3/5/6 days; eval = J20+7 days (Jan 27).
Graph growth from eval histories: 3,489 entries on a 1,009,701 log (+0.3%) for `eval`; the training folds get much larger additions (f1 +70,505 ≈13%, val23 +20,250) → train/eval mismatch, the likely reason B ≤ A.

## Validation (local, same settings; baseline = matching v12 component retrained/stored with identical recipe)
Baseline source: v12 stored models `v12/ens/dwbig_c{cut}_s0.npy` (v12x/xlog `lgb255` reproduces C25 .81658 exactly with `xlgb.py`); recipe dwbig = LightGBM lr .015, 127 leaves, ff .5, λ2 30, min_data 200, max_bin 255, 1000 rounds, drop `^sq_(pdot|pcos|mdot|mcos)$`, features feat_lr2,tt,pp,sqC,dw. New runs add 25 columns (435→460 features), nothing else changes. Seed noise of v12 baseline ≈ ±.00006 (seed0 .81658 vs seed1 .81652 at C25).

dwbig-recipe LightGBM alone (AUC):
| cut | v12 baseline | A (train-only) | B (TRANSDUCTIVE) |
|---|---|---|---|
| C25 | .81658 | .81719 (+.00061) | .81704 (+.00046) |
| C23 | .79460 | .79631 (+.00171) | .79633 (+.00173) |
| C26 | .82520 | .82599 (+.00079) | .82581 (+.00061) |

Full v12 blend at C25 (dwbig + dw_s + 2×MLP×3 seeds, raw-logit mean; all components retrained with the new features):
| | dwbig | dw_s | MLP(3) | blend |
|---|---|---|---|---|
| v12 (stored) | .81658 | .81595 | .81650 | .81723 |
| A | .81719 | .81675 | .81715 | .81802 (+.00079) |
| B | .81704 | .81672 | .81702 | .81793 (+.00070) |
C23/C26 blends were not retrained (only dwbig). Eval candidates: Spearman B vs A 0.99967; B vs v12 0.99619.
Leak check (`test_leakB.py`): rebuilding B's val25 features after flipping every label with ts≥cutoff → bit-identical; stored feat_B/val25 matches the rebuild.

## Candidates (`out/`; not leaderboard-uploaded; local C25 blend .818 < .83 gate so not packaged by `tools/package_final.py`)
- `submission_v17cooc_B_TRANSDUCTIVE.csv.zip` sha256 `45dba8b8bb49f74506088aa948f272cecc003ff9643b583f91f8dadd35a5a5c6`
- `probe_v17cooc_B_TRANSDUCTIVE_p33_w80.csv.zip` (0.8 rank model + 0.2 rank p33) sha256 `3fd6f639ee27563bf26ffd0e34cd6c29e43b02ba43835af8f2891d24b968af3e`
- (comparison) `submission_v17cooc_A_trainonly.csv.zip` `45b62455ee826632c49775704eca1b832cfeb50b188e045e9b78d22b86c08b38`; `probe_v17cooc_A_trainonly_p33_w80.csv.zip` `e35f01258462b70ebdd5e196771d73db46e9fabd82b46010f45d72be51cc526d`
All: 529,390 rows, rowid set == v5, header `rowid,predicted_probability`, no NaN, in [0,1].

## Runtime (8 threads, no GPU)
Feature build: ~13–22 s per table (10 tables per variant, ≈3 min; `eval` 22 s). Cut-27 training: dwbig 4.8 min, dw_s 1.4, MLP 3×1.6 → ≈11 min.

## Reproduction
```
cd /workspace/huawei/v17cooc
python mkcooc.py B f1 f2 f3 f4 f5 f6 val23 val25 val26 eval    # (A likewise) -> feat_B/*.parquet (needs dw_p33/p33lib.py, v12/mkdw.py read-only)
python mksel.py                                                  # -> feat_Bp / feat_Ap (25 selected columns)
./run_lgb.sh                                                     # dwbig-recipe LGB, cuts 25/23/26, A and B  (xlgb17.py)
./run_blend.sh                                                   # dw_s + MLP x3 (+dwbig c27) for B c27, B c25, A c25, A c27
python assemble.py B --out ; python mkprobe17.py submission_v17cooc_B_TRANSDUCTIVE   # blend AUC, csv, zip+sha256, p33 0.8/0.2 probe
python test_leakB.py
```
`cl.py`, `xlgb17.py`, `ens17.py`, `mlp17.py` are copies of v12/v12x loaders that also look for feature dirs in v17cooc/.

## Risks
- B ≈ A − .0001: eval histories add only +0.3% to the graph at eval time, while training folds saw +13% (mismatch); gain is entirely from the low-k coordinate/cos/prod features, not from the transductive part.
- Gain (+.0007 blend at C25) is small versus public/private noise (±.0005); expected leaderboard effect ≲ +0.0007.
- B violates the evaluation-aggregate ban except as explicitly approved for this feature.
- 4-core/15-minute limit not addressed: adds ≈20 s per table + SVD at inference time.
