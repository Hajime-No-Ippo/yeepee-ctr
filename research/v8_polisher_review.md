# v8 review: Spec Polisher (2026-10-23, ~00:10 IST)
Inputs read: research/v7_error_analysis.md, research/v8_plan.md, INSTRUCTIONS.md, v7_err_scripts/err.py + err2.py, v5/report.md (+ out/decide_vC.log), v2work/out/timings.jsonl.
No jobs run; nothing in v2work or v5 modified.

## 0. Two blockers before any v8 work

1. **The 15-min / laptop-CPU / no-GPU rule is not in INSTRUCTIONS.md.** It has no runtime, hardware or code-submission clause. The rule is presumably from the webinar or the organisers. **Eric should confirm the source and exact wording**, e.g. whether it is end-to-end from raw CSVs and on which machine.
2. **v7 lite_nox2 is already over 15 min.** The measured run (`v2work/out/timings.jsonl`, run 23:39, 4 threads pinned) totals **1006 s = 16.8 min**:

| Stage | Wall (s) |
|---|---|
| feat_v2 | **454** |
| feat_v3 | 161 |
| fit_predict_slow | 91 |
| fit_predict_base | 80 |
| feat_lr | 75 |
| feat_x | 62 |
| feat_tt | 47 |
| load / write / validate / blend | ~36 |

`lite_summary.md`, which the plan cites as the runtime reference, does not exist; the timings jsonl is the only source.
Every "+1–3 min" item in the plan therefore needs runtime freed first. **feat_v2 (45% of wall time) is the lever.** Dropping v2 entirely cost −0.0035 (nov2), so profile it per block instead of removing it (spec S0).

## 1. Bottleneck causes: does the evidence support them?

| Claim | Verdict | Why / likelier explanation |
|---|---|---|
| **B1: low-activity users (d0–d2) are the biggest deficit (~0.017)** | **Partly.** The segment is real; the size is overstated. | (a) The deficit metric `eval_share × (overall − segment AUC)` isn't how global AUC aggregates. A segment's internal AUC excludes cross-segment pairs, and B1/B2/B3 overlap (d0 contains every empty-history and most history-only users), so 0.017 + 0.008 + 0.005 double-counts. (b) d0's 0.72 is partly the empty-history rows (AUC 0.55) sitting inside it; re-cut d0 without empty users. (c) The eval/val share gap (33.5% vs 22–29%) is **selection, not drift**: val25/26 only contain users with an event in the last 1–2 train days (active users), while eval contains ~every user. That makes it genuine for eval, and it is why the reweighted AUC is the honest number. |
| **B2: long-gap / stale users (~0.008 + unmeasured tail)** | **Supported, but mostly the same population as B1, and partly mechanical.** | In eval, ≥1 d since last activity is largely *implied* by staleness: any event ≥1 d after the cutoff has ≥1 d gap, because nothing after C is visible. The lastact-mix reweighting (−0.0022 / −0.0049) is the one measured number and it is real. The ≥2 d tail (16.8% of eval) is untestable on val25/26, so **only val23 (0–4 d stale) can measure it**. A joint activity × lastact table is needed before B1 and B2 get separate work. |
| **B3: cold / empty-history (~0.005, 3.4% of eval)** | **Cause supported; the ceiling is low.** | With no history and often no train rows, only item-side and time signals exist, so ~0.55 is close to what item popularity can give. Upside is bounded by 1.3% of rows, so this needs a cheap fix, not a tower. |
| **"Within-event ranking is 2/3 of the error"** | **The numbers are right; the framing isn't.** | The swap oracle (+0.155) and the event-label oracle (+0.073) are *headroom under perfect information*. They aren't additive shares of error, and neither is reachable. In marginal terms the event-level side has looked cheaper per AUC point but has repeatedly failed in practice (v4 event/choice stacking +0.001; eventmodel stacking +0.0003), while within-event embedding work keeps paying (v5 vA +0.0065, vC PPMI +0.0044–0.0114). So the *conclusion* (invest in within-event / choice features) is supported by the experiment history more than by the oracle split. Note: err.py's first within-oracle is lexicographic on the event score; the table correctly reports err2's swap oracle. |
| **Monday shift is untestable** | **Supported.** | Every val target is Thu–Sun, and the Monday rows in eval are exactly its <1 d stale rows, so day and staleness are confounded. val23's Thu/Fri rows are the closest proxy. Also note that v2work's chosen `notime` variant *dropped* time/day features because they hurt. Plan #6 / C.1–C.3 re-introduce day features and must re-earn that against notime. |
| Item tail is not a test bottleneck | **Supported.** | Eval is head-heavy (46.8% in top 3 popularity deciles). |

## 2. v8 experiments vs INSTRUCTIONS.md and Eric's rules
INSTRUCTIONS.md data rule: stats/labels from the fitting part only (ts < C); keep an event's rows together. Eric's rule: no aggregates over unlabelled eval rows; train data + each eval row's own history only.

| # | Item | Data rules | Runtime / CPU | Format / other | Verdict |
|---|---|---|---|---|---|
| 0 | val23 + eval-mix gate | OK (eval used only for segment *shares*, no labels, no features) | 0 at submission | – | **Go.** Must exist in v2work before anything else is gated. |
| 1 | PPMI port (`v5/feat_pp`) | OK: ppmi.py builds from CLg only, with an assert (verified) | +1–2 min, must be timed | – | **Go.** Gain verified in v5/out/decide_vC.log vs vA: +0.0064 / +0.0044 / +0.0114 (val25/26/23). Expect less on v7 (two-tower overlap). |
| 2 | Stale folds + weighting + gap features | OK if folds use ts < C | Bigger folds raise fit time | – | **Demote.** Eval-like folds already failed twice (v4 s2 half-sample; v2work `eft` 0.77749 vs 0.77748). The +0.002–0.005 estimate has no supporting run. Only retry the *weighting* on existing e/E folds, gated on val23 ≥2 d. |
| 3 | Low-activity / cold tower | OK (cluster priors from train-only Ug) | +1 min | – | **Reshape** into S2 (shrinkage features), not a separate model. |
| 4 | NN listwise member (torch CPU) | OK (no GPU needed) | +2–3 min claimed, unmeasured; cap rows and threads | **A per-event softmax output can't be used as the submission score** (it removes between-event signal and kills global AUC). Blend its pre-softmax logit only. | Go **after** S0 frees time. |
| 5 | LGBM tuning + lambdarank member | OK | Tuning ±0; ranker +1–2 min | Early stopping on val25 only, report on val26/23. lambdarank alone lost in v3 (0.7485 vs 0.7556), so blend member only. | Low priority. |
| 6 | Day-of-week features | OK (train-only CTRs). Train has only 2 weekend days, so smooth heavily | ~0 | Conflicts with the `notime` finding | Hedge only. Accept if val23 weekday doesn't drop **and** val25/26 don't drop. |
| 7 | xAn clean columns | OK | +1 min for +0.0003 | – | **Drop** under the 0.0005/min admission rule. |
| A | Adversarial validation | Diagnostic only. Any drop list is Eric's call: it is driven by eval feature distributions. | Dev only | – | OK as written. |
| B | GBDT+LR, torch FM | OK | +1–3 min each | Blend output must be mapped back to **[0,1]** (sigmoid of blended logit). `logit_mean + w·within-event z` likewise → sigmoid. Global rank-averaging must be scaled to [0,1]. | GBDT+LR cheap and fine; FM after S0. |
| C | Missing cross-CTRs C.1–C.7 | OK if every count/label is from ts < C and frozen at C in validation; m≈20 smoothing | ~1 min | C.1–C.3 are day/time features (see #6) | C.4, C.5, C.6 are the B1-relevant ones. Do those first. |
| D | uwin multi-window user stats | OK: windows end at the event ts and use only rows with ts < C (val targets are first events ≥ C, so nothing sits in between) | ~30 s | – | Go; it is cheap and directly separates "gone quiet" from "always sparse" (B1 vs B2). |

Submission-format checks for whatever ships: header `rowid,predicted_probability`, 529,390 unique rowids matching evaluation.csv, finite values in [0,1], UTF-8, no index column. Run `v3/validate_submission.py` plus the v2work validator. Nothing gets uploaded without Eric.

## 3. Specs for the next push (in order)

### S0: runtime headroom (blocker)
- Profile `feat_v2` by sub-block (co-click tables, SVD16/24, ui repeat, item CTR windows) with `timed_run.py` on one table (eval).
- Rank blocks by gain share in the v7 base model (`ens/*_model*.txt`, `feature_importance('gain')` summed per v2 block).
- Options in order:
  1. Vectorise or cache the slowest block across folds (same cutoff → same stats).
  2. Drop v2 blocks with <0.5% total gain.
  3. Restrict feat_v2 to folds f3–f6 + eval (lite already uses f3–f6 for training).
- **Acceptance:** lite end-to-end ≤ **12.5 min** on the same 4-core pinned setup (leaves ~2.5 min for S1 + S3), with val25/val26/val23 each within −0.0005 of v7 lite.

### S1: PPMI port (B1 + within-event)
- Copy `v5/ppmi.py` read-only into the v8 work dir. Point paths at the v8 dir and build `feat_pp` for f3..f6, val23, val25, val26, eval. Join via `cl.matrix(extra=('feat_pp',))`.
- **Before building:** Pearson of `x_pp_*` vs `feat_tt` (two-tower) columns on val25. If max |corr| > 0.9, keep only the non-overlapping columns.
- **Acceptance:** CLg-only assert passes; time it (+≤2 min); screen gate Δ val25 ≥ +0.0005 and Δ val26 ≥ 0; full 3-seed confirm ≥ +0.001 on val25/26/23.

### S2: low-activity shrinkage + uwin (B1, B2)
Build the joint table first (seconds, from saved val25/26 preds + a v7 val23 run): AUC by `act_dec (d0–d2 / d3–d5 / d6–d9) × lastact (<1d / ≥1d) × stale (val23 only: <1d / 1–2d / ≥2d)`, with empty-history rows split out. Spend effort only on cells with ≥3% eval share and AUC ≥0.01 below overall.

Features (new `v8/lowact.py`, train-only statistics):
- `x_con05_k32_shr`: the user contrast vector shrunk toward its KMeans-256 cluster centroid (KMeans fit on train-only `Ug`) with weight `n/(n+m)`, n = the user's click count, m ∈ {3, 10}. Plus `_z`.
- `x_hist_coclk_max` / `_mean`: max/mean train-only co-click count (or PPMI if S1 lands) between the candidate and each history item. Skip if `v2_co_cnt` already equals it (check corr > 0.95).
- C.4 user-cluster × item-popularity-bin CTR; C.6 item × user-activity-bin CTR. m=20 smoothing toward item CTR, counts from ts < C.
- `uwin.py`: 1 h / 1 d / 7 d user impressions, clicks, CTR, and the 1 d / 7 d ratio. About 12 columns.

**Acceptance:**
- d0–d2 segment AUC up ≥ +0.003 on val25 and val23.
- Overall gate as S1.
- Total added runtime ≤ 1.5 min.

### S3: within-event blend members (structural)
- **GBDT+LR:** leaf indices of the base model on lite training rows. Then `LogisticRegression(C tuned on val25, solver='saga')`, logit-averaged with weight w ∈ {0.1, 0.2, 0.3}. Final score = sigmoid(blend).
- **Blend tweak:** `logit_mean + w·z_within_event` with w tuned on val25 and checked on val26/val23; output = sigmoid.
- **NN listwise:** only if S0 leaves ≥2 min. Cap at 1M rows and 4 threads, and blend its pre-softmax logit.
- **Acceptance:** each member ≥ +0.0005 per extra minute on all three val sets; final file passes both validators.

### S4: B3 empty-history (cheap)
- Single-feature AUC inside the empty segment (val25/26/23) for item CTR, item × time_bucket CTR and item recent-6h CTR. If the full model is already at that ceiling, stop.
- Otherwise add `us_nohist × it_ctr` interaction columns, no separate model.
- **Acceptance:** empty-segment AUC ≥ +0.02 with overall not worse.

### S5: Monday hedge (last)
C.1 item × day_type CTR (heavily smoothed: only 2 weekend train days) and a weekday flag.
**Acceptance:** val23 weekday segment not worse **and** no loss on val25/26 vs notime. Otherwise drop it. This guards the 66.6% Monday share; it isn't a gain play.

### Deprioritised
- #2 stale folds: failed twice; retry only weighting, gated on val23 ≥2 d.
- #7 xAn: fails the runtime admission rule.
- Tuning sweep: after S0–S3.

## 4. For Eric (via Grok Bot)
1. Confirm the exact source and wording of the 15-min laptop-CPU rule: what is timed (raw CSV → submission?) and on what machine. INSTRUCTIONS.md doesn't contain it, and v7 lite already measures 16.8 min here.
2. Adversarial-validation drop lists stay proposals until he approves them.
