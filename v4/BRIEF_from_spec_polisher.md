# v4 build plan (Spec Polisher, 2026-10-23 03:47 IST). Start ONLY after v3 final fit + submission_v2.csv are done.
Reasoning and code sketches: /workspace/huawei/research/auc_ideas_v1.md (§1 lists the v3 gaps, §4 has implementation notes).

## Ground rules
- Work in /workspace/huawei/v4/. Copy or import from v3; never edit v3, because it is the fallback submission.
- Every earlier rule still holds: label stats frozen at cutoff C (assert P_rows.ts.max() < C), final scores are binary LightGBM with no per-event softmax or rank, CPU only, seeded, no uploads.
- Build no feature from the unlabelled eval rows (e.g. candidate counts during the eval period). Eric hasn't decided whether that's allowed.
- Base is the 164-feature v3 set. Add groups one at a time and record each in v4/out/ablation.json.

## Step 1: honest validation (first)
- Pick the best iteration on val25 and REPORT val26 (iteration count scaled to the data). Log val25 too; val26 is the headline.
- Add a val22 set: cutoff 22 Jan, targets are each user's first event from the cutoff to the end of train.
- AUC by staleness bucket (<=1d, 1-2d, >2d), plus eval-weighted AUC with w = eval_share[bucket]/val_share[bucket] passed as sample_weight. In eval, 33.4% of rows are more than 1 day stale.
- Event-level AUC: sum of p per event vs whether the event has any click. v3 baseline is about 0.649 (val25) and 0.656 (val26).
- Re-score the v3 best config under this protocol for a fair comparison.

## Step 2: folds that match eval (idea B), v4/build_feats_v4.py
- Targets: each user's first event in [C, 27 Jan) (W=None), giving 0-6 days of staleness like eval.
- Half-day cutoffs J20 + k*DAY/2 for k >= 2, only cutoffs before the validation cutoff.
- *_per_day version (count / days in fit window) for every count column: it_imp, it_clk, it_nclk_all, us_nev, us_nclk_win, hist_imp, and the rest.
- Final fit must stay at about 15 minutes or less; row-sample by event if not.
- Report whether it helped, using the Step 1 metrics.

## Step 3: user taste vector (idea A), after the SVD block
- Keep the k=64 SVD item vectors V fixed. Per user: w_u = (V^T W V + lambda I)^-1 V^T W y over P_rows impressions (clicked=1, skipped=0, weight 1) plus the click log (y=1, weight 1 + a*count).
- Grid: lambda in {1, 10}, a in {0, 2}. Chunks of about 5k users.
- Features: fi_l{lambda}_a{a}_dot = w_u.v_cand and its within-event z-score; fi_cold flag (w = 0 for users with no data).
- Check: single-feature AUC on val25 vs sv_k64_con05 (0.709 alone).

## Step 4: event model + choice model (idea C)
- Event model: one row per target event, label = any click.
  - Features: us_*; clicked in each of the last 1/3/5 events before C; time since last event and last click; event count in the last 24h; tm_*; event-level max/mean/std of con05, hcos and the fold-in score; count of candidates already in history.
  - Binary LightGBM; out-of-fold predictions grouped by user give p_event.
- Choice model: lambdarank on clicked events only, giving s_choice. p_share = within-event softmax(s_choice), used as a FEATURE only.
- Add p_event, p_share and p_event*p_share to the main binary model.
- Report event AUC vs 0.649/0.656, and global AUC with and without these three features.

## Deliverable
- Only if the best v4 config beats v3 on BOTH the val26 report-only AUC and the eval-weighted AUC: write /workspace/huawei/submission_v3.csv with the same validator. Otherwise write no new submission.
- Report back with:
  - v3 vs v4 table: val22, val25 (early stopping), val26 (reported), eval-weighted AUC, event AUC, staleness buckets
  - ablation for each step
  - top 15 features
  - final-fit runtime
  - submission path, if one was written
- Out of scope for now: iALS/item2vec, repeat-click model, Otto co-visitation, SASRec, blending with submission_v1.
