# v12 NOTES (v11 + DW feature families)
## 21:10 step 1: features
- mkdw.py: reusable.all_recommended (42 cols: impression_taste+slate_centrality+skip_contrast) for rows of f1..f6, val23/25/26, eval -> feat_dw/*.parquet. Fit cutoffs: f_k=J20+k*DAY, val23/25/26, eval=J27 (train ts<X only). Eval: eval events appended to a Data copy only for own-history lookups; snapshot is exactly train events (assert). 1453 eval users unseen in train -> empty user row (NaN features).
- val25 features reproduce DW cache exactly (42 cols, 0 mismatches). ~60-75s per cut.

## step 2: method (b) full retrain (LightGBM dwbig+dw_s, MLP dwmlp x3 seeds) -- done
b_full = dwbig:1 + dw_s:1 + 2*dwmlp:3 (seeds 0,1,2), same recipe as v11
 v11: C25 .81524  C23 .79424  | b_full: C25 .81723 (+.00199)  C23 .79566 (+.00142)
 LGB-only swap (v11 MLP kept): C25 .81668, C23 .79523
 timings: LGB dw_s ~1min, dwbig ~4min per cut; MLP ~3-4min per run (see timings.jsonl)

## step 3: method (a) residual boost (resid.py oof, 5-fold grouped by user, seeds 0,1) -- done
 C25 +.00152/+.00156 vs v11 ; C23 +.00131/+.00129 ; C26 +.00156/+.00156   (matches DW's +.00146/+.00143)
 => (b) full retrain wins on all three cuts: C25 +.00199, C23 +.00142, C26 +.00170 (C26 on MLP seed 0 only: .82604 vs .82434).
 Chosen: (b). Second-seed confirmation: LGB seed 1 for dw_s/dwbig and v11 sC_s/sCbig on cuts 25/23 (b3.sh); MLP per-seed comparison.

## step 4: SUBMISSION BUILT (method b, seed-0 recipe) 21:45 Dublin
 spec: dwbig:1 + dw_s:1 + 2*dwmlp:3 (seeds 0,1,2) at cut 27; raw logit mean -> sigmoid
 out/submission_v12.csv: 529,390 rows, rowids == v8 set, unique, p in [0.00026,0.831], no NaN. Spearman vs v11 .9938, vs v8 .9809, vs v10 .9773
 gate: C25 .81723 / rw .81722 < .83 -> refused (exit 1) as expected
 candidate zip: out/submission_v12_candidate.csv.zip sha256 1dcf21846cf7fa8c7de356f630d7888b6ff4f739299c544354df57f6d6d16506
 csv sha256 37898b6f51715f4aa19400366545aee5da715332fce5ccfe3b3f421229c4b10b
 NEXT: second-seed confirmation (b3.sh, LGB seed 1 dw vs v11 baselines, cuts 25/23). b3.sh was never launched (shell aborted).

## step 5: second-seed confirmation (LGB seed 1; cuts 25/23) -- PASS 21:~ Dublin
 v11 (LGB s1, MLP s1)        C25 .81523 C23 .79424 | v12b (LGB s1, MLP s1) C25 .81721 C23 .79557  (+.00198 / +.00133)
 v11 (LGB s1, MLP s0-2)      .81529 / .79423       | v12b                  .81727 / .79562        (+.00198 / +.00139)
 v11 (LGB s0+1, MLP s0-2)    .81529 / .79429       | v12b                  .81728 / .79568        (+.00199 / +.00139)
 accept rule (C25 up, C23 not down >.0002) holds for both seeds. Submission uses seed-0 LGB + 3 MLP seeds (not the 2-seed LGB average).
 runtimes (sum of wall): DW feature builds ~534s, dw_s 361s, dwbig 1425s, dwmlp 599s(+prep), seed-1 baselines 415s, resid oof ~2min. Whole v12 ~1h45 incl. waits.
 scratch cleaned (__pycache__); kept feat_dw, ens, logs.

## PACKAGING TASK (v12 -> sol/solution_v12.py) 00:50 Dublin
- Plan: generator sol_work/make_sol12.py builds sol/solution_v12.py from v10 sol (SRC dict embedded) + seq11 (SQ_SEED/H/HID) x11 seeds (0-7 base; 8-10 SQ_H=4 SQ_HID=128) + avg_sq + DW (common/signals/reusable/mkdw, paths->W) + ens (dw_s, dwbig) + mlp11 x3 seeds + blend (dwbig+dw_s+2*mlp mean ep4)/4.
- Workdir for verification: /tmp-like clean dir with only train.csv/evaluation.csv symlinked; keep output stats in sol_work/verify_stats.json

## PACKAGING progress (00:57 Dublin)
- Built generator sol_work/make_sol12.py (+head12.txt, tail12.txt, src12/ modules) -> sol/solution_v12.py (no /workspace paths; stages incl. feat_sq (seqrun.py: V12_SQ_SEEDS default 11, par 3) + feat_dw + dwbig/dw_s/mlp0-2/blend/write/validate; --cut 25 = local val mode).
- dwbig trained 1000 rounds (not 1200) since blend only uses round 1000; first 1000 trees identical.
- Verify run 1 (clean/raw symlinks only train.csv+evaluation.csv, work=sol_work/clean/work): stages setup..feat_pp, log clean/run1.log. Then compare matrix vs v12 cache, then --from feat_sq.
- 01:13 stages setup..feat_pp done 15.4 min (feat_tt 635 s). feat_sq+feat_dw running -> clean/run2.log

## 01:40 Dublin - BOX RESTART incident
- Box rebooted ~01:20 (uptime 15 min at 01:35, journald 'uncleanly shut down'); /workspace was re-hydrated from a snapshot (all mtimes 01:20-01:27) WITHOUT any directory named `venv` or `out` (every out/ folder is gone: v12/out/submission_v12.csv, v2work/out/..., etc). Not caused by my clean test (only touched v12/sol_work/clean).
- venv rebuilt with uv at /workspace/venv (py3.13.5, pinned pkgs from sol/requirements.txt + pandas/pyarrow/duckdb/matplotlib).
- My verification run 2 (feat_sq) was killed by the restart (msC/s0 only partial). Features from stages setup..feat_pp survived in sol_work/clean/work.
- v12 shipped CSV regenerable from ens/{dwbig,dw_s,dwmlp}_c27 npy files (blend 1:1:2).
- 01:50 v12/out/submission_v12.csv RESTORED byte-identical (sha 37898b6f...10b) via recov/regen.py from ens/ npy (dwbig c1 + dw_s c1 + 2*mean dwmlp c3)/4, sigmoid, evaluation.csv order, header rowid,predicted_probability. Seq verification run 3 relaunched 01:43 (--from feat_sq --to feat_dw), 11 seeds.
- 01:38 probe built: probes/probe_v12_p33_w80.csv.zip sha 798cf0f0...7562; spearman(v12,p33)=0.9291
- 02:23 clean run: feat_sq done in 2525s (11 seeds; eval cols corr .99-1.0 vs shipped sqC); feat_dw running
- 02:27 clean run: feat_dw 467s, eval cols corr 1.0 vs shipped. Launching --from dwbig --to validate (run3.log)
- 02:28 v11/out/submission_v11.csv regenerated from ens npy (sha f8bdd0a4..b0cb; no original full sha on record -> unverified; Spearman vs v12 .9938)
- 02:28 tools/package_final.py REF -> data/evaluation.csv (one-line change), gate re-tested: format pass, tampered rowid fails, low-AUC fails. Lost out/: v8 sub, v3 sub, v10 sub (sha bee2c7d5.. on record; from timed solution run, not regenerable w/o rerun), v13 (unshipped), v5/out results.jsonl+SEL.json, candidate zips.
- 02:42 CLEAN RUN c27 COMPLETE: out/submission_v12_sol.csv valid; Spearman vs shipped v12 = 0.99943 (pearson .99931, max|dp| .0507, mean .09350 vs .09335). Stage times sum 4635s = 77.3 min (8 threads): v3 prep 36, feats v3/v2/x/lr 220, tt 635, pp 35, seq(11 seeds) 2525, dw 467, dwbig 321, dw_s 74, mlp x3 321, blend/write/val 7. Fixed bug: --out dir now created (makedirs). stats in sol_work/verify_stats_c27.json. Launching --cut 25 in sol_work/clean25 (run1.log).
- 02:47 README_run.md written (c25 run in progress; prop features: mkprop.py ported at v12/prop/dw/mkprop.py, tables built in clean/work/feat_prop and clean25/work/feat_prop; matches DW cache_prop_c25 except tie-order in p_l3/p_streak for 0.03% rows)
- [03:52 UTC] c25 clean run DONE: LOCAL AUC cut25 dwbig .81661 dw_s .81611 mlp .81646 blend .81727 (v12 ref .81723, +.00004). Stats in sol_work/verify_stats_c25.json. Next: XGBoost check in /workspace/huawei/v12x (Eric request) BEFORE propensity.
