# Runtime-cut spec: v17cooc (variant A) -> v20fast, <15 min (target <=12.5 min) on 4 cores, no GPU
Author: Spec Polisher. Spec only; no production code. Read-only inputs: v17cooc, v17run, v12, v12x, v11, v10, research/lite_summary.md. All writes go to /workspace/huawei/v20fast.
Tags: **M** = measured, **E** = estimated (reasoning stated), **S** = needs screen (no evidence).

## 0. Headline
Rebuilding the whole v17cooc from raw CSVs on 4 cores is **about 65-75 min (E)**. That is 5-6x over the target. The only measured 4-core run under 15 min is v10 (727.9 s, AUC C25 .81388, `v10/NOTES.md`), which is .0041 below A (.81802). The decision rule below will most likely end with a **frontier report**, not a pass. Lean model stage and fewer seq seeds are cheap in AUC. Getting under 12.5 min also forces the seq and dw feature stages to be cut or removed, and that costs real AUC.

## 1. Stage-time budget, variant A, cut 27 (final fit)
8-thr = v12 clean run `v12/NOTES.md` (sum 4635 s, `verify_stats_c27.json`) plus `v17cooc/ens/ens_log.jsonl` for the co-occurrence and model stages. 4-core = lite_nox2 timings (`research/lite_summary.md`, taskset 0-3, M) where they exist, otherwise 8-thr x factor (E). Factors: polars features x1.4 (lite 4-thr feat_v3 161 s vs 8-thr 113 s), LGB x1.9, torch MLP x1.5, seq x1 per seed (torch barely scales: 1 thr 69 s, 4 thr 54 s, `v10/NOTES.md`).

| Stage | 8-thr s | Source | 4-core s | Basis |
|---|---|---|---|---|
| prep_v3 + prep_v2 | 36 | M c27 | 28 | M (lite 18.5+9.3, warm cache; cold +90 s) |
| feat_v3 + v2 + x + lr | 220 | M c27 | ~416 | 161+118+62+75; v2 118 is the clean figure (454 was a stall, lite_summary) |
| feat_tt | 635 | M c27 | ~47 | M lite 4-core 47; c25 run 35 s. The 635 is likely contention, **re-time** |
| feat_pp | 35 | M | ~45 | E |
| feat_sq, 11 seeds | 2525 | M c27 (c25: 1800) | ~1800 | E: 11 seeds, 1 thr each, 4 in parallel = 3 waves x ~600 s |
| feat_dw | 467 | M c27 (c25: 304) | ~600 | E |
| cooc A build (5 tables + SVD) | ~90 | E from README (13-22 s/table) | ~120 | E |
| dwbig (1000 rd, 460 feat) | ~225 | M `ens_log` coAp c27 | ~420 | E |
| dw_s | 69 | M | ~130 | E |
| MLP x3 | ~255 | M (1.4-1.5 min each) | ~380 | E |
| blend + write + validate | 7 | M | 8 | E |
| **Total** | **~4560 s (76 min)**; 3960 s with tt=35 | | **~3990 s (~67 min)** | E |

Not measured: no timed v17cooc run exists on 4 cores. If `v17run/work/c27/timings.json` exists when Code Buddy starts, use it as the 8-thr baseline. Other agents' jobs keep the box at load ~9, so re-time quiet.
Serial floor: everything except seq and models is ~1250 s (E). Under 12.5 min it must run concurrently across cores. CPU/wall in lite_nox2 ranges 1.0 (feat_v2) to 2.9 (tt), so packing helps.

## 2. Ranked cut levers
Evidence figures are LightGBM or blend AUC deltas on local cuts C25/C23.

| # | Lever | Time saved (4-core E) | AUC cost | Evidence |
|---|---|---|---|---|
| 1 | **Model stage lean: dwbig + dw_s + MLPx3 -> dwbig + MLP x1** | ~380 s (dw_s + 2 MLPs). Using dw_s + MLP x1 instead saves ~670 s | C25 **+.00004** (b+mlp1 .81806). dw_s+mlp1 .81791 (-.00011). b+s+mlp1 .81799. mlp3 vs mlp1 .81715 vs .81716 | M: my blend subsets of saved A preds `v17cooc/ens/co*_c25_s0.npy`, 11-seed features. **C23 not measured: S** |
| 2 | **Seq seeds 11 -> 4** | ~1100 s (but 4 seeds still fill 4 cores for ~600 s) | C25 -.00019, C23 -.00033 (sAbig .81441/.79276 vs sCbig .81460/.79309) | M `v11/NOTES.md`. 8->11 seeds only +.0002 C25 |
| 2b | Seq seeds 11 -> 1 (1 pinned core, overlapped with features) | ~1650 s total; ~0 on the critical path if hidden behind features | C25 **-.00054, C23 -.0022** (big1 .81406/.79092 vs sCbig). **Fails the C23 rule** | M v11. Seeds 2-3 are **S** |
| 3 | **Drop fold f3 (nfolds 4 -> 3 everywhere: v3, v2, x, tt, dw, cooc, seq, LGB rows)** | ~20% of every per-table stage, ~250-350 s plus ~20% of fit time | v7 lite: -.00005/-.00018 (other pipeline) | Pipeline-specific, so **S** on v17 |
| 4 | dwbig negfrac 1.0 -> .5 | ~35-40% of dwbig fit (~150 s) | b/s pair at nf1 vs .5: +.00006 / +.00002, i.e. no gain | `v11/NOTES.md` b1. dwbig itself **S** |
| 5 | dw trim (keep best families, drop others) | up to ~300-400 s | Full dw is worth +.0020 C25 / +.0014 C23 (v12 vs v11). Per-family value unknown | **S** |
| 6 | Drop seq entirely | all of it | v10B vs v8: seq stack ~+.0007; 11 seeds ~+.0014 total (E). Unmeasured on v17 | **S** |
| 7 | feat_tt / feat_v2 anomalies | 0-590 s | none | Re-time only |
| 8 | Pinning: taskset 0-3, all thread vars 4, `SQ_THREADS=1`, `V12_SQ_PAR`=seeds, `num_threads=4` | tens of s, avoids thrash | none | v10: unpinned overlap thrashed (feat_x 427 s vs 56) |
| 9 | Cooc: build only the 25 selected columns | ~40 s | 0 | **S** |
| ✗ | seq maxsteps 6000 -> 3000 | - | -.0005/-.0003 (raw .782 -> .755) | Rejected, v10C |
| ✗ | seq ep2 bs512 | -30% fit | rejected (24-48h -.0006) | Rejected, v10E |
| ✗ | max_bin 63/511/1023, lambdarank, xendcg, CatBoost, XGB | none or worse | -.0001 to -.008 | v12x, v7 lite |

## 3. Frontier ladder to measure (all E until measured)
| Config | 4-core wall (E) | C25 (E) |
|---|---|---|
| L0 v17cooc A | ~67 min | .81802 (M) |
| L1 = L0 with dwbig + MLP x1 | ~61 min | .81806 (M) |
| L2 = L1 + seq 4 seeds | ~45 min | ~.8178 |
| L3 = L1 + seq 1 seed pinned, overlapped, nfolds 3 | ~20-25 min | ~.8165 (S) |
| L4 = dw_s + MLP x1, nfolds 3, no seq, dw kept | ~13-15 min | ~.815-.816 (S) |
| L5 = L4 with dw trimmed | ~10-12 min | S |

## 4. Steps for Code Buddy (one heavy job at a time; every heavy command starts with `taskset -c 0-3 env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 POLARS_MAX_THREADS=4 RAYON_NUM_THREADS=4 TT_THREADS=4 MLP_THREADS=4 SQ_THREADS=1`)
- **S0 setup (light).**
  - `mkdir /workspace/huawei/v20fast`. Copy `v17run/{v17run.py,run_v17cooc.sh,requirements.txt,src}`. Copy `v17cooc/{xlgb17.py,ens17.py,mlp17.py,cl.py,assemble.py}` into `v20fast/screen/`. Never edit originals.
  - Wait for v19/v17run jobs to finish (`pgrep`, load <2). Check: `diff -r` against the originals shows only intended edits.
- **S1 model-side screens on cached features, C25 and C23.**
  - Features: `v12/feat_lr2,tt,pp,sqC,dw` + `v17cooc/feat_Ap`. Retrain on C23 any member you need.
  - Configs: M0 reproduce b+s+mlp3 (.81802 C25); M1 b+mlp1; M2 s+mlp1; M3 b (negfrac .5)+mlp1; each also with `--nfolds 3`.
  - Output `v20fast/screen_models.jsonl` with fit seconds **at 4 pinned cores**.
  - Check: M0 within .0001 of .81802; the model times replace the E column in section 1.
- **S2 feature-side screens (use the best S1 recipe, M2 for speed).**
  - Variants: no feat_sqC; `v10/feat_sq10` (1 seed); no feat_tt; each dw family dropped via `--drop` regex (families named in `src/dw/reusable.py`); no cooc.
  - Check: table of C25/C23 deltas per feature block with its 4-core build seconds, so each family has AUC per second.
- **S3 4-core feature timings.**
  - Run `python v17run.py --variant A --cut 27 --threads 4 --to feat_pp --work v20fast/work`, then `feat_dw` and `cooc_A` separately.
  - Check: replace E values with M; record `timings.json`; flag any stage with CPU/wall ~1 (candidate for lane overlap).
- **S4 lane scheduler `v20fast/run_fast.py`.** Reads the dependency of each stage from the files it opens.
  - Seq lane: `taskset -c 3`, `SQ_THREADS=1`, `V12_SQ_SEEDS=k`, `V12_SQ_PAR=1` (starts after prep_v3 only). Feature lane: independent stages on cores 0-2.
  - Models run after both lanes, on 4 cores.
  - Check: no stage starts before its inputs exist; per-lane time logged.
- **S5 candidate runs.** Run the 2-3 best configs from the ladder end to end.
  - Raw CSVs only, cut 27 (time) and cut 25 (C25 AUC), variant A. C23 AUC comes from the S1/S2 cached-feature recipe, since v17run has no val23 table.
  - Check per config: wall seconds (taskset 0-3, all thread vars 4), C25, C23, delta vs A.
- **S6 submission validation (cut 27 of the chosen config).**
  - Header `rowid,predicted_probability`; 529,390 rows; rowid set == `data/evaluation.csv`; no duplicates; finite in [0,1].
  - Final score = `sigmoid(raw-logit blend)`. No per-event softmax or rank.
  - Compare Spearman vs `v17cooc/out/submission_v17cooc_A_trainonly.csv`. Do not upload.
- **S7 report**: `v20fast/REPORT.md` with the frontier table (config, 4-core wall M, C25, C23) and the winning config.

## 5. Decision rule
Accept the **fastest** config with total wall <=750 s (12.5 min) on 4 pinned cores AND C25 >= .81802 - .0015 = **.81652** AND C23 not worse than A's by >.0015.
If none passes, report the frontier. Give the best config under 15 min (900 s) and the best AUC under each of 12.5/15/20/30 min. Do not relax the AUC bounds on your own.
Every number must be M, not E. Re-time when load >2. Differences below .0005 are noise (public/private split).

## 6. Data rule
Variant A only: graph and all stats are fit on train events with ts < cut. No eval-row aggregates. B (transductive) is out unless Eric approves. Any new dw or cooc trim must not add aggregates over unlabelled eval rows.

## 7. Open questions for Eric
1. Does the 15-min limit cover **raw CSV -> submission** (features + training + inference), or only inference with pre-trained models/artifacts? If only inference, most of this spec is moot: only eval-table features + predict remain, a few minutes.
2. Exact laptop: physical cores or hyperthreads, CPU generation, **RAM** (a seq seed needs ~2 GB; the box has 15 GB), SSD vs HDD, cold vs warm cache. Is `train.csv` available at run time?
3. Is "<15 min" a hard disqualifier, and how much AUC is he willing to give up, given the .0015 bound likely fails at 12.5 min?
4. May Code Buddy run quiet-box timing runs of ~15-30 min each while v19/v17run are idle?
5. Is a pinned lane layout (seq on 1 core, features on 3) acceptable for the official run, or will the harness run a single process?
