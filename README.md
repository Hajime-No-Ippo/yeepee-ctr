---
library_name: lightgbm
license: other
metrics:
- roc_auc
pipeline_tag: tabular-classification
tags:
- ctr
- lightgbm
- tabular
- recsys
- click-through-rate
- learning-to-rank
- huawei-techarena
model-index:
- name: yeepee-ctr
  results:
  - task:
      type: tabular-classification
      name: CTR prediction (10 candidates per event)
    dataset:
      name: Huawei TechArena 2026 Phase 1 (private competition data, not included)
      type: huawei-techarena-2026-phase1
      split: local-validation-cutoff-25jan
    metrics:
    - type: roc_auc
      value: 0.81321
      name: ROC AUC, local validation (cutoff 25 Jan, first event per user after cutoff)
      verified: false
  - task:
      type: tabular-classification
      name: CTR prediction (10 candidates per event)
    dataset:
      name: Huawei TechArena 2026 Phase 1 (private competition data, not included)
      type: huawei-techarena-2026-phase1
      split: public-leaderboard
    metrics:
    - type: roc_auc
      value: 0.814419
      name: ROC AUC, public leaderboard (submission_v8)
      verified: false
---

# Yeepee — Huawei TechArena 2026 CTR (Phase 1)

Private team repo for our click-through-rate solution to the Huawei TechArena 2026 (Ireland) CTR challenge.
Each event shows 10 candidate items, and we predict a click probability per row. The score is **global ROC AUC** over all rows.
It contains **code, logs, reports, metrics and small model dumps only**. There is no competition data, and no feature caches, predictions or submissions.


<img width="7395" height="7212" alt="diagram" src="https://github.com/user-attachments/assets/70f38a6d-8b04-459b-bed3-bcef6c4c6475" />


**Current best: v8**, public AUC **0.814419** (v7: 0.814396). Local AUC: 0.81321 (cutoff 25 Jan), 0.82188 (26 Jan), 0.79056 (23 Jan).
v8 = mean of two LightGBM models (base and slow) on 377 features, blended 0.8/0.2 in logit space with a DIN-style sequence model.

## Where things live
- `v8b/sol/solution_v8.py`: **the shipped v8 pipeline**. It is a single self-contained driver that embeds all of its modules. `v8b/src/` holds readable copies of those modules.
- `v8/`: v8 research. This covers the feature builders (`lrfeat2.py`, `ttfeat.py`, `ppmi.py`, `seq.py`), `v8_report.md` and the screen reports (`screens_report.md`, `screens_v8c.md`).
- `v9d/`: public-gap diagnostics after v8 (`report_v9d.md`, `diag.py`, `val/`). These cover drift, day-of-week, adversarial reweighting, train-window, all-negatives and plain LR member screens.
- `v2work/out/submission_log.csv`: every submission with its sha256 and local and public AUC.
- Older: `v2work/`, `v3/` (v3 pipeline), `v4/`, `v5/`, `parallel*/`, `research/` (idea notes).

## How to run v8
You need the competition files, which are not in this repo. A laptop CPU is enough (~10 GB RAM, ~25 GB disk); the run takes about 626 s end-to-end on 4 cores.
```bash
python v8b/sol/solution_v8.py --raw /path/to/raw_csvs --work ./v8_work --threads 4 --out ./v8_work/submission_v8.csv
# The raw dir must contain train.csv and evaluation.csv. Stages: prep -> v3/v2 fold features -> extras/lr/tt/ppmi/seq
# -> base + slow LightGBM at cutoff 27 Jan -> blend -> write + validate (also writes the .zip).
# You can resume with --from STAGE / --to STAGE.
```
Validation screens use the lite setup in `v9d/val` (`scr.sh TAG feat_lr2,feat_tt,feat_pp "25 26 23"`, then `cmp.py TAG`).

## Data rule: no evaluation-row aggregates
- Every global, item-level or co-occurrence statistic (popularity, CTR, co-clicks, transitions, SVD/MF factors, clusters, sequence-model training) uses **training events before the cutoff only**.
- An evaluation row may use **its own user history** (for that row's own user features), and nothing pooled across evaluation rows.
- `submission_v2` was withdrawn for breaking this rule. Treat it as a hard rule for any new feature.

## Validation protocol
Time-based cutoffs (23, 25 and 26 Jan) score the first event per user after the cutoff, with all statistics frozen at the cutoff and each event's 10 rows kept together.
Local and public results don't line up exactly. Small local gains (≤ 0.0015) have not reliably shown up on the public board; see `v9d/report_v9d.md`.

## License
`license: other`. The code is the team's competition work. The competition data belongs to the Huawei TechArena 2026 organisers and is not redistributed; the competition rules apply.
