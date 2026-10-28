# v14 — TRANSDUCTIVE TEST VERSION: uses aggregates over unlabelled evaluation rows (normally banned by Eric; enabled only for this version)

**Status: NO SUBMISSION BUILT. No transductive feature showed a gain beyond noise, so v14 is not worth submitting.** (`submission_v14_TRANSDUCTIVE.csv` was deliberately not produced; nothing was uploaded.)

## Evaluation-row statistics that were tested (all label-free)
Code: `feat_tx.py::build_tx` (copy of /workspace/huawei/dw_lb/feat_tx.py). Target rows T = validation first-event rows (each user's first event at/after cutoff C, 10 candidates each; C=25 → Jan 25-27, C=23 → Jan 23-27). For a submission the same function would be run over the 52,939 evaluation.csv events (slates, ts, histories). n = number of target events. "j" = candidate item, i = the row's event. Own slot/own history is removed (leave-one-out). Train-side reference quantities use only train events with ts<C and their labels (already allowed).
No label of any target-window row is used by any feature (only slate items, event ts, history items/ts).

| family | feature | formula | aggregated over |
|---|---|---|---|
| f1 | f1_cnt | (#target events whose slate contains j − 1)/n·1e4 | all target slates |
| f1 | f1_cnt_h1/h6/h24 | same count restricted to target events within ±1h/6h/24h of ts_i, divided by number of target events in that window | target slates |
| f1 | f1_late_exp, f1_late_ctr, f1_ratio_late, f1_ratio_all, f1_exp_x_ctr | target exposure ÷ train exposure of j in last day (and all train); late-train CTR (ts<C, labels of ts<C) | target slates + train ts<C |
| f1 | *_z, *_dm, *_rk, f1_slate_mean | within-event z-score / demeaned / rank, slate mean of f1_cnt | within the event |
| f2 | f2_cooc_norm / f2_cooc_raw | mean over the other 9 candidates k of (cooc_T(j,k) − 1)/sqrt(cnt_j cnt_k), cooc_T = co-appearance count in target slates | target slates |
| f2 | f2_lastshare, f2_lastshare_n | among other target users whose LAST history click equals i's last click h: share whose target slate contains j | target slates + target histories |
| f2 | f2_l3share | same using i's last 3 history clicks | target slates + target histories |
| f3 | f3_hpop, f3_hpop_l3, f3_hpop_l1, f3_hpop_excl_own, f3_own_cnt | count of j in the (last 20 / recency-weighted last 3 / last 1) history clicks of target users, per 1e4 users | target histories |
| f3 | f3_hpost, f3_hrecent_6h/24h, f3_hnext_6h/24h, f3_hpost_vs_train_ctr | count of j in target histories with ts≥C / in the window before or after ts_i | target histories |
| f4 | f4_trel_h, f4_trank, f4_n_same_h, f4_n_6h, f4_n_before_frac, f4_hist_n | hours since C, rank of ts among target events, #target events within ±0.5h / ±6h, history length | target event timestamps |

## Validation result (residual LightGBM on v12 logits, init_score, user-grouped 5-fold CV, control = z-derived context only, event-bootstrap 95% CI, 150 resamples)
Metric = global AUC; Δ = AUC(test) − AUC(control). v12 validation AUC: C25 0.81723, C23 0.79566. Null noise feature: Δ = −0.00006 (C25), 0.00000 (C23).

| set | C=25 seed0 | C=25 seed1 | C=23 seed0 |
|---|---|---|---|
| f1 exposure (20) | −0.00002 [−0.00014,0.00015] | −0.00005 [−0.00019,0.00009] | +0.00008 [−0.00007,0.00026] |
| f2 co-occurrence (14) | +0.00013 [0.00004,0.00023] | +0.00012 [0.00003,0.00022] | −0.00005 [−0.00014,0.00004] |
| f3 history popularity (20) | +0.00010 [−0.00005,0.00022] | +0.00013 [−0.00001,0.00028] | +0.00009 [−0.00005,0.00023] |
| f4 timing (6) | +0.00003 | +0.00005 | +0.00007 |
| all transductive (60) | +0.00010 [−0.00008,0.00024] | +0.00009 [−0.00010,0.00026] | +0.00012 [−0.00004,0.00028] |
| all transductive + user propensity (88) | +0.00018 [−0.00001,0.00033] | +0.00017 [0.00001,0.00033] | +0.00029 [0.00013,0.00047] |

Univariate check on C25 val rows: every transductive feature has within-event AUC ≤ 0.53 (best: f3_hpop 0.532, almost all of it the item-in-own-history effect, f3_own_cnt 0.551); f3_hpost (post-cutoff clicks) 0.500; timing 0.500. The synthetic data is stationary and exposure is unrelated to clicks.
The most that could be claimed is ~+0.0001 (all transductive) vs a +0.0002–0.0003 from the non-transductive user-propensity features alone, i.e. the transductive part adds nothing measurable. A leaderboard gain is therefore expected to be below the public/private noise (≈±0.0005), and well short of 0.83.

## Reproduction
```
cd /workspace/huawei/dw_lb
/workspace/venv/bin/python feat_tx.py 25,23        # builds cache_tx_c{25,23}.npy (needs dw_signals Data loader)
/workspace/venv/bin/python run_tx.py 25,23 0       # per-family residual test -> results_tx.jsonl
```
(Copies of the scripts are in this folder; run from dw_lb because they import dw_levers/dw_signals read-only.)

## Risks if someone did build it
- Window mismatch: validation windows are 2 days (C25) / 4 days (C23) but evaluation spans 7 days (Jan 27–Feb 3, 70% of events in the first 2 days), so count statistics would shift; they would have to be normalised per event (they are) and training folds would have to use matching windows.
- Public/private are random halves of evaluation.csv, so transductive statistics over ALL evaluation rows move both halves equally — no public-vs-private selection effect, but also no leak-driven protection against overfit.
- Training folds need mirrored statistics (per pseudo-cutoff fold over its own first-event rows); that rebuild is expensive and unjustified for ≤+0.0001.
- Violates the evaluation-aggregate ban unless explicitly accepted for this version.
