# solution_v12.py - Huawei TechArena 2026 Phase 1 (CTR), end-to-end from the raw host files

One self-contained script: raw `train.csv` + `evaluation.csv` -> all features -> models -> `submission_v12.csv`
(columns `rowid,predicted_probability`, 529,390 rows, values in [0,1]). No absolute paths: everything lives under `--raw`, `--work`, `--out`.
All module sources are embedded in the script and written into `--work` at start-up.

## What it does
1. v8/v10 feature families (v3 + v2 + extra similarity + lr2 + tt + pp), 377 columns.
2. `feat_sqC`: DIN-style sequence model refit at every fold cutoff and averaged over **11 seeds** (env `V12_SQ_SEEDS`), 20 stacking columns.
3. `feat_dw`: DW impression-taste (SVD of the user x item impression matrix), slate-centrality and skip-contrast features, 42 columns.
4. Models on folds f3..f6 (435 columns): LightGBM `dwbig` (1000 rounds), LightGBM `dw_s` (500 rounds), 3-seed MLP; logit blend 1:1:2, then sigmoid.
5. Writes and validates the CSV (row count, unique rowids, [0,1], no NaN).

## Requirements
* Linux x86_64, **Python 3.13** (3.11+ should work), CPU only. `pip install -r requirements.txt`
  (torch: `pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu`).
* About 12 GB RAM, about 40 GB free disk in the work dir, 8 cores recommended.
* Data: put `train.csv` (7.2 GB) and `evaluation.csv` together in one directory (the `--raw` dir). Nothing else is needed.

## Run
```bash
python -m venv venv && . venv/bin/activate && pip install -r requirements.txt
python solution_v12.py --raw /path/to/data_dir --work /path/to/scratch --threads 8 --out /path/to/submission_v12.csv
```
Stages are resumable: `--from STAGE --to STAGE` (stage list in the script docstring; finished stages are timed in `WORK/timings.json`).
Run it with `setsid nohup ... &` for the long run.

Env knobs: `V12_SQ_SEEDS` (default 11; fewer seeds = faster and slightly weaker), `V12_SQ_PAR` (concurrent sequence runs, default min(3, threads//2)),
`V12_SQ_THREADS` (torch threads per sequence run, default 2). Set `OMP/MKL` thread vars yourself only if you know why; the script sets them per stage.

## Expected runtime (8 threads, 8-core box, 15 GB RAM)
Verified end to end from a work dir holding only the two raw CSVs: **77.3 min** total (sum of stage times; the box restarted once mid-run, the sequence stage was redone in full).

| stage | seconds |
|---|---|
| prep (v3 + v2) | 36 |
| feat v3 / v2 / x / lr | 220 |
| feat_tt | 635 |
| feat_pp | 35 |
| feat_sq (11 seeds, 3 concurrent x 2 threads) | 2525 |
| feat_dw | 467 |
| dwbig / dw_s | 321 / 74 |
| mlp seeds 0,1,2 | 126 / 95 / 100 |
| blend + write + validate | 7 |

The sequence stage dominates (about 33 % of the time at 11 seeds); `V12_SQ_SEEDS=3` cuts it to about 12 min.

## Verification result
* Output vs the shipped `submission_v12.csv`: Spearman **0.99943** (Pearson 0.99931, mean p 0.09350 vs 0.09335); seed noise only.
* Local check mode: `--cut 25` runs the same recipe on folds f1..f4 and scores `val25` (first event per user in Jan25-Jan27, labels from train); it prints/saves the AUC
  in `WORK/local_auc_c25.json` and does not write a submission. Use a separate `--work` dir per cut. Shipped v12 local AUC at C25 = 0.81723.
