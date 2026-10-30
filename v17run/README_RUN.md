# v17run - rebuild v17cooc from the raw data (Yeepee, Huawei TechArena 2026 CTR)

`v17cooc` = the verified **v12** recipe (435 columns, LightGBM `dwbig` + `dw_s` + 3-seed MLP, logit blend 1:1:2)
plus **25 item co-occurrence columns** (`mkcooc`: per-user time-ordered, minute-deduplicated click log -> window W=20, weight 1/d item x item counts
-> PPMI (alpha 1) -> randomized SVD (24 comp.) -> user vector = centroid of the item vectors of the row's own history; cosine / z / dm variants) = 460 columns.

| variant | what the co-occurrence graph is built from | files |
|---|---|---|
| **A** | train events only (`ts < fit cutoff`) | `submission_v17cooc_A_trainonly.csv` |
| **B - TRANSDUCTIVE** | train events **plus the history entries of the evaluation events** (no labels exist for them; own-history only). For the evaluation table only. | `submission_v17cooc_B_TRANSDUCTIVE.csv` |

Variant B uses unlabeled evaluation inputs when fitting a feature extractor; treat it accordingly when you decide what to submit.
The public best probe `probe_v17cooc_B_TRANSDUCTIVE_p33_w80` (0.818302) is `0.8*rank(B) + 0.2*rank(p33)`. p33 is a teammate's model whose ranks are a file we
cannot regenerate here, so that last step is **optional** (`--p33 P33_PATH` / env `P33_PATH`, a parquet with columns `rowid, p33_rank`).

## Layout
```
v17run/
  run_v17cooc.sh     entry point (checks packages, calls v17run.py)
  v17run.py          driver: ordered, resumable stages, config, thread env
  requirements.txt   pinned versions (Python 3.13.5, Linux x86_64, CPU)
  README_RUN.md
  src/               pipeline sources (the verified v12 stage scripts + src/cooc/ = co-occurrence step, blend, probe, compare)
```
`src/` holds the sources with two placeholders (`@@WORK@@`, `@@RAW@@`); the first stage (`setup`) writes the substituted copy into `WORK_DIR/c<cut>/`
and every stage runs there (cwd = that folder), so all module imports are plain sibling imports inside one folder and there are **no absolute paths** in the repo.
Do not run the scripts under `src/` directly - always use `run_v17cooc.sh`.

## Setup
```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt            # torch: pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
```
Needs ~16 GB RAM, ~50 GB free disk in `WORK_DIR`, 8 cores recommended (`--threads`).

## Data layout (nothing is shipped in the repo)
```
$DATA_DIR/train.csv          # raw TechArena training log (rowid,event_id,user_id,item_id,timestamp,time_bucket,day_type,history_item_ids,history_timestamps,clicked)
$DATA_DIR/evaluation.csv     # raw evaluation log (same columns without clicked), 529,390 rows
```
`DATA_DIR` defaults to `./data`, `WORK_DIR` to `./work` (relative to the v17run folder); override with env vars or `--data/--work`.

## Commands
```bash
export DATA_DIR=/path/to/csvs WORK_DIR=/path/to/scratch
./run_v17cooc.sh                          # both variants, cut 27 -> submissions   ($WORK_DIR/c27/out/)
./run_v17cooc.sh --variant A              # only A (train-only)
./run_v17cooc.sh --variant B              # only B (TRANSDUCTIVE)
./run_v17cooc.sh --cut 25                 # local validation (folds f1..f4, scored on val25 with the train labels) -> $WORK_DIR/c25/local_auc_c25_{A,B}.json
./run_v17cooc.sh --p33 /path/p33_recovered_ranks.parquet   # also writes the probe_* files (optional, needs teammate file)
./run_v17cooc.sh --list                   # print the stage order
./run_v17cooc.sh --from A_big --to A_blend --force   # redo a range
```
Stage order (resumable; each finished stage drops `WORK_DIR/c<cut>/.done/<stage>`, rerun the same command to continue; timings in `timings.json`):
`setup prep_v3 prep_v2 feat_v3 feat_v2 feat_x feat_lr feat_tt feat_pp feat_sq feat_dw cooc_A cooc_B` then per variant V
`V_big V_s V_mlp0 V_mlp1 V_mlp2 V_blend [cut 27: V_write V_validate (V_probe)]`.
The driver never uploads anything; submit the CSV/zip by hand.

Outputs (cut 27): `$WORK_DIR/c27/out/submission_v17cooc_{A_trainonly,B_TRANSDUCTIVE}.csv` (+`.zip`, `rowid,predicted_probability`), `$WORK_DIR/c27/timings.json`.
`python src/cooc/compare.py NEW.csv REF.csv` prints Spearman / Pearson between two submissions.

## What is in each stage
- `prep_*, feat_v3, feat_v2, feat_x, feat_lr, feat_tt, feat_pp`: v12 feature families (fit cutoffs f_k = Jan20 + k days, k = 3..6, eval at Jan27; all aggregates use `ts < cutoff`).
- `feat_sq`: DIN-style sequence model refit at every cutoff, 11 seeds averaged (the slowest stage; `V12_SQ_SEEDS=3` is a faster, weaker option and breaks equality with the reference).
- `feat_dw`: 42 impression-taste / slate-centrality / skip-contrast columns.
- `cooc_A/B`: `mkcooc.py` (58 columns) then `mksel.py` keeps the 25 used by v17cooc (`feat_Ap/`, `feat_Bp/`).
- `V_big` LightGBM (lr .015, 127 leaves, 1000 rounds); `V_s` LightGBM (lr .03, 95 leaves, 500 rounds, 50% negatives); `V_mlp0-2` 512-256 MLP x 3 seeds; `V_blend` `p = sigmoid((big + s + 2*mean(mlp))/4)`.

## Reproducibility
See the table at the bottom (filled in from the clean rebuild test). Nondeterminism sources: the 11-seed sequence stack (torch, thread-count dependent float order), LightGBM multithreading, MLP threading.
So a rebuild is **statistically equivalent, not bit-identical**, to the shipped v17cooc files.
