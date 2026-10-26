# Yeepee v5 — running report (Code Buddy)
Brief: BRIEF_from_spec_polisher.md. Approved by Eric 23 Oct 15:02 IST. All results append to out/results.jsonl.

## Step 0 — setup
- 15:03 IST: v2work manifest taken (v2work_manifest_before.txt, 521 files). Code + needed tables copied (feat: h1-5,val25,val26,eval; feat_v3/feat_x/feat_x2: f1-6,val25,val26,eval; data parquet + evaluation.csv).
- Path patches: `rg -n v2work v5/*.py` -> empty. combo.py num_threads=3, screen.py --threads default 3.
- val23 added (SPEC in v3fold/extra/extra2, c23 in build_folds, V2_OF in combo/cl, screen efold .get). Built 15:04-15:06: val23 501,550 rows in all four families.
- Runner: v5val.py (same featlist/params as combo.py val; seeds 42/43/44; 400 rounds; logs per-seed + mean).

### Acceptance 0 — PASS (15:34 IST)
| cut | AUC@400 mean (42/43/44) | per seed | event AUC | within-event | warm | hist-only | empty | weekday | weekend | stale<1d | stale>=1d |
|---|---|---|---|---|---|---|---|---|---|---|---|
| val23 | 0.73672 | .7371/.7367/.73636 | 0.62793 | | 0.74177 | 0.69419 | 0.52295 | 0.74077 | 0.70723 | 0.74695 | 0.71192 |
| val25 | 0.77766 | .77748/.77781/.7777 | 0.65403 | | 0.77972 | 0.72353 | 0.56456 | – | 0.77766 | 0.78207 | 0.7589 |
| val26 | 0.79346 | .79355/.79342/.79342 | 0.66430 | | 0.79476 | 0.72361 | 0.5435 | – | 0.79346 | 0.79346 | – |
- val25 mean 0.77766 vs 0.7775 target: +0.00016 (tolerance ±0.0005). Seed 42 = 0.77748, identical to v2work ne_combo25. 325 features. Saved out/base.json.
- Waiting on Grok Bot's ens.py (started ~15:35) before A0 heavy jobs.

### A0 notes
- out/a0check_noflag_invalid.log (`A0CHECK {}`) is NOT a result: the first a0check ran without FIX=1, so nothing was captured. extra.py now forces FIX in a0check mode and asserts iC/iN/km64/km256 were captured. Re-run queued in runA1.sh.
- base_fix (feat_x rebuilt with FIX=1: iC/iN/KMeans fit from train-only Ug) val25 3-seed mean 0.77769 vs base 0.77766 (+0.00003) -> within −0.0005: PASS. Base stays the comparison base (no material move).

### A0 identical-check (item 4): PASS
I built eval with T_ev=EE/ER and again with EE.head(0)/ER.head(0) (FIX=1). Max |diff| was 0.0 for Ug k32/k64/k128, iC, iN, km64 and km256 (out/a0check.log, 16:22).

### A1 sweep acceptance: PASS
The ref config k32/tfidf/b0.5/a0.5 reproduces x_con05_k32 with corr 1.0 and within-event AUC 0.72573 (target 0.7257 +/- 0.002).

### A1 pass 1: single-feature table (val25, 48 configs, sorted by within-event AUC)
| name | auc_raw | auc_z | within | corr_ref |
|---|---|---|---|---|
| x_con05_k32_bm2505 | 0.71877 | 0.72693 | 0.72591 | 0.9819 |
| x_con05_k32_tfidf05 | 0.71836 | 0.72602 | 0.72573 | 1.0 |
| x_con05_k32_bm250 | 0.72132 | 0.72615 | 0.72474 | 0.973 |
| x_con05_k32_tfidf0 | 0.72059 | 0.72495 | 0.72449 | 0.9916 |
| x_con05_k32_bm251 | 0.71149 | 0.72449 | 0.72391 | 0.9704 |
| x_con05_k32_tfidf1 | 0.71118 | 0.7238 | 0.72388 | 0.987 |
| x_con05_k24_tfidf0 | 0.71964 | 0.72481 | 0.7237 | 0.7449 |
| x_con05_k32_logidf0 | 0.71965 | 0.72497 | 0.72362 | 0.9536 |
| x_con05_k32_logidf05 | 0.71633 | 0.72503 | 0.72353 | 0.9617 |
| x_con05_k24_bm250 | 0.71837 | 0.72373 | 0.72249 | 0.744 |
| x_con05_k48_bm251 | 0.71102 | 0.72345 | 0.72239 | 0.7763 |
| x_con05_k16_tfidf0 | 0.71803 | 0.72315 | 0.72235 | 0.7184 |
| x_con05_k24_tfidf05 | 0.71513 | 0.72338 | 0.72217 | 0.7627 |
| x_con05_k48_tfidf1 | 0.71083 | 0.72309 | 0.72171 | 0.7776 |
| x_con05_k32_logidf1 | 0.70841 | 0.72188 | 0.72108 | 0.9492 |
| x_con05_k48_bm2505 | 0.71549 | 0.72246 | 0.72108 | 0.7544 |
| x_con05_k24_bm2505 | 0.71389 | 0.7223 | 0.72101 | 0.7605 |
| x_con05_k48_logidf1 | 0.70871 | 0.72165 | 0.72101 | 0.7724 |
| x_con05_k48_tfidf05 | 0.71529 | 0.72205 | 0.7209 | 0.7547 |
| x_con05_k16_bm250 | 0.71484 | 0.72054 | 0.71999 | 0.7143 |
| x_con05_k48_logidf05 | 0.71389 | 0.72149 | 0.71982 | 0.7536 |
| x_con05_k24_logidf0 | 0.71498 | 0.72069 | 0.7197 | 0.7382 |
| x_con05_k24_tfidf1 | 0.7064 | 0.71894 | 0.71833 | 0.7644 |
| x_con05_k16_tfidf05 | 0.71145 | 0.71935 | 0.71831 | 0.7277 |
| x_con05_k24_logidf05 | 0.70997 | 0.71874 | 0.71799 | 0.7525 |
| x_con05_k24_bm251 | 0.70537 | 0.71792 | 0.71685 | 0.7614 |
| x_con05_k48_bm250 | 0.71457 | 0.71823 | 0.7167 | 0.7194 |
| x_con05_k48_logidf0 | 0.71382 | 0.71805 | 0.71644 | 0.7212 |
| x_con025_k32_bm2505 | 0.70924 | 0.71714 | 0.71642 | 0.9596 |
| x_con05_k48_tfidf0 | 0.71418 | 0.71775 | 0.71629 | 0.7185 |
| x_con05_k16_bm2505 | 0.70824 | 0.71664 | 0.71598 | 0.722 |
| x_con05_k24_logidf1 | 0.70107 | 0.71385 | 0.71361 | 0.751 |
| x_con05_k16_logidf0 | 0.70803 | 0.71409 | 0.71303 | 0.7049 |
| x_con05_k16_tfidf1 | 0.70135 | 0.71281 | 0.71226 | 0.7221 |
| x_con05_k16_bm251 | 0.69836 | 0.71014 | 0.70992 | 0.7153 |
| x_con05_k16_logidf05 | 0.70106 | 0.70982 | 0.70911 | 0.7101 |
| x_con05_k16_logidf1 | 0.69105 | 0.70302 | 0.70282 | 0.701 |
| x_con05_k48_bin05 | 0.69168 | 0.70254 | 0.70146 | 0.751 |
| x_con05_k48_bin0 | 0.69575 | 0.70297 | 0.70129 | 0.7331 |
| x_con05_k48_bin1 | 0.68313 | 0.69876 | 0.69763 | 0.7517 |
| x_con05_k32_bin0 | 0.6886 | 0.69667 | 0.6961 | 0.8971 |
| x_con05_k32_bin05 | 0.68224 | 0.69341 | 0.69309 | 0.8887 |
| x_con05_k32_bin1 | 0.67248 | 0.68704 | 0.68681 | 0.8592 |
| x_con05_k24_bin0 | 0.67646 | 0.68461 | 0.68397 | 0.7006 |
| x_con05_k24_bin05 | 0.66947 | 0.67997 | 0.67969 | 0.6989 |
| x_con05_k24_bin1 | 0.6599 | 0.67285 | 0.67256 | 0.6822 |
| x_con05_k16_bin0 | 0.65263 | 0.66025 | 0.66002 | 0.6292 |
| x_con05_k16_bin05 | 0.64599 | 0.65525 | 0.65489 | 0.6227 |
| x_con05_k16_bin1 | 0.63777 | 0.64846 | 0.64864 | 0.6048 |

### A1 selection (pass 1 + pass 2, 60 configs)
I took the top configs by within-event AUC that have |corr| < 0.9 vs x_con05_k32 and vs each other:
x_con05_k24_tfidf0 (within 0.7237, corr_ref 0.745), x_con1_k32_bm251 (0.72257, 0.856), x_con05_k48_bm251 (0.72239, 0.776), x_con05_k16_tfidf0 (0.72235, 0.718).
The configs that beat the ref (k32 bm25 b0.5, 0.72591) are near-duplicates (corr ≥ 0.97), so I excluded them. No single config meaningfully beats the ref; the picks were chosen for diversity. Written to out/SEL.json.

### Status 17:07:35
feat_xA build is queued behind another agent's pipeline (solution_v4_lite.py in /workspace/huawei/lite, writing to v2work/out/submission_v4_lite.csv). guard.sh is waiting on it. That pipeline writes to v2work, so the v2work manifest will differ for reasons outside this run.

### Status 17:08:09
The feat_xA build is queued behind another pipeline (solution_v4_lite.py, run from /workspace/huawei/lite, writing v2work/out/submission_v4_lite.csv). That pipeline writes into v2work, so the v2work manifest will differ through no action of this run.
17:07: per Grok Bot's request, guard.sh now also waits on any python process with cwd or args under v2work. I will not build the binary last-30-40-clicks SVD k12 1/sqrt(pop+5) feature (Grok Bot v6).

### feat_xA build acceptance: PASS
All 10 tables: row count and rowid set match feat_x, 0 NaN/inf in the new columns. 31 new columns (A1 12, A2 12, A3 7). Removed x_con05_k64 (+_z/_rk) for A4.

### Step A 25%-user screens (frac 0.25, 2 seeds, 300 rounds, f-folds; XDIRS feat_xA,feat_x2)
| tag | n_feats | val25 | Δ25 | val26 | Δ26 | pass |
|---|---|---|---|---|---|---|
| base_v5 (feat_x_fix minus dup k64) | 322 | 0.77368 | – | 0.79050 | – | – |
| scr_A1 (4 sweep picks +_z/_rk) | 334 | 0.77775 | +0.00407 | 0.79419 | +0.00369 | PASS |
| scr_A2 (item side k32/k64) | 334 | 0.77787 | +0.00419 | 0.79444 | +0.00394 | PASS |
| scr_A3 (taste confidence) | 329 | 0.77567 | +0.00199 | 0.79210 | +0.00160 | PASS |
All three groups pass. Event AUCs on the screens are 0.613–0.620. A control screen (ctl_fix = feat_x_fix, 325 feats) is queued to confirm base_v5 is not an unlucky low.

### Step B pre-check: STOP
sl_cand_cent_cos on val25: within-event 0.50601 (flipped 0.49399), global 0.50706, 36 s. Both directions are ≤ 0.55, so I did not build Step B.
| ctl_fix (control, feat_x_fix 325) | 325 | 0.77378 | +0.00010 | – | – | control OK |
| scr_Astack (A1+A2+A3) | 353 | 0.78148 | +0.00780 | 0.79721 | +0.00671 | PASS |

The stack passes. Full 3-seed confirmation (vA: feat_xA,feat_x2, all of A1–A3, 353 feats) started 18:20 on val25, val26 and val23.

### Status 18:24:24
vA confirmation is running. val25 seed 42: 0.78428 (base seed 42: 0.77748). Queued: runFinal.sh (decision -> combo final with XFDIR=feat_xA -> write_sub -> both validators -> Spearman), then runC.sh (C1 PPMI check -> build -> screen).

## Decision rule: vA (A1+A2+A3, 353 feats) vs base (3-seed means, 400 rounds)
| cut | model | AUC | Δ | event AUC | stale<1d | stale≥1d | weekday | weekend | runtime s |
|---|---|---|---|---|---|---|---|---|---|
| val23 | base | 0.73672 |  | 0.62793 | 0.74695 | 0.71192 | 0.74077 | 0.70723 | 379 |
| val23 | vA | 0.74626 | +0.00954 | 0.63109 | 0.75649 | 0.72131 | 0.75039 | 0.71584 | 337 |
| val25 | base | 0.77766 |  | 0.65403 | 0.78207 | 0.7589 | – | 0.77766 | 564 |
| val25 | vA | 0.78437 | +0.00671 | 0.65721 | 0.78891 | 0.76506 | – | 0.78437 | 591 |
| val26 | base | 0.79346 |  | 0.6643 | 0.79346 | – | – | 0.79346 | 744 |
| val26 | vA | 0.79995 | +0.00649 | 0.66736 | 0.79995 | – | – | 0.79995 | 831 |

DECISION PASS: Δ val25 +0.00671, val26 +0.00649, val23 +0.00954 (all ≥ +0.001). val23 weekday 0.74077 -> 0.75039 (+0.00962).

Top-15 gain importances for vA:
- val23: x_con05_k48_bm251_z 0.0715, ev_hist_n_all_z 0.0389, x_icos_k32_z 0.0377, x_icon025_k64_z 0.0309, x_con05_k24_tfidf0_z 0.0291, x_icos_k64_z 0.028, ev_sv_k64_con05_z 0.026, v2_uip_n_dm 0.0247, x_icon1_k32 0.0205, ev_hist_n_all_rk 0.017, x_icon05_k32_z 0.0167, x_icon1_k32_z 0.0161, v2_svd16_cos_dm 0.0146, x_icon05_k64_z 0.0122, x_con05_k16_tfidf0_z 0.0115
- val25: x_con05_k48_bm251_z 0.1769, ev_hist_n_all_z 0.0371, x_icon1_k32_z 0.0338, x_icon05_k64_z 0.0335, x_con05_k24_tfidf0_z 0.032, x_icon1_k32 0.0293, x_icon05_k32_z 0.0245, x_icon025_k64_z 0.0236, x_con05_k16_tfidf0_z 0.0218, x_icos_k32_z 0.0185, ev_sv_k64_con05_z 0.0182, x_con1_k32_bm251_z 0.0171, x_con1_k32_bm251 0.0152, ev_hist_n_all_rk 0.0135, v2_svd16_cos_dm 0.0132
- val26: x_con05_k48_bm251_z 0.2077, x_icon1_k32_z 0.0455, x_con05_k24_tfidf0_z 0.0375, x_icon1_k32 0.0336, x_icon05_k64_z 0.0323, x_con05_k16_tfidf0_z 0.0296, x_icon05_k32_z 0.0257, x_con1_k32_bm251_z 0.0235, x_con1_k32_bm251 0.018, ev_sv_k64_con05_z 0.0164, x_icon05_k64 0.0159, r_l3n 0.0153, sv_k64_con05 0.0151, v2_svd16_cos_dm 0.0142, x_icos_k32_z 0.013

### Status 19:01:07
Decision PASS. runFinal is waiting in guard.sh for Grok Bot's v6 ens.py (v2work) before it starts combo final. runC (C1) is queued after FINAL_DONE.

## v2work manifest (before = v2work_manifest_before.txt, after = out/v2work_manifest_after.txt, diff = out/v2work_manifest.diff)
v2work changed, and every change came from other agents' pipelines (Grok Bot's v4/v4_lite/v6/lr/ideas runs), not from v5. Evidence: `rg -n v2work v5/*.py` is empty. v5 reads only from v5/ref copies (rc_val25, seq_c25), which were copied once for the C1 correlation check. guard.sh logged those agents' processes (ens.py v6base/v6slow --lr feat_lr2, extra.py, solution_v4_lite.py, lrfeat2.py, ideas_q*/lr_q* screens) running there throughout.
- Added: cache_mx/* (~70 parquet+json), ens/{base,estrc,slow,wide}_c27_*, ens/v6base_c25/26_*, ens/v6slow_c25_s0.npy, ens/queue27.*, ens/seq_analysis.json, feat_lr/*, feat_lr2/*, feat_xf/*, feat_rc/{E1..E6,eval}.parquet, eval_pred_v4.parquet, extra.py.bak_leak, ideas_q1/q2.{sh,log}, ideas_v2_screen_results.txt, lite_build/*, lr_q1/q2.{sh,log}, lrfeat.{py,log}, lrfeat2.py, mods.py, out/run1_*, out/solution_v4*.py/zip, out/solution_v6.py, out/submission_v4.csv(.zip), out/submission_v4_lite.csv(.zip), scr_*_c25/c26.npy (~50), showr.py, ttfeat.py, v6_run.{sh,log}, __pycache__/mods.pyc
- Modified: cl.py, ens.py, extra.py, screen.py, screen_results.jsonl, ens/blend_results.json, ens/ens_log.jsonl, out/submission_log.csv (18:44; last row is "submission_v4.csv.zip (public score) … 0.789237"), __pycache__/cl.pyc
- Removed: seqdbg.py

## Final v5 (decision PASS)
- combo final: XF=1 XF2=1 XFDIR=feat_xA, 400 rounds, 3 seeds, notime; 353 feats, 2,105,880 train rows. Ran 19:19:44–19:31:42 (about 12 min, 3 threads). Final top features: x_con05_k48_bm251_z 0.2303, x_icon1_k32_z 0.0521, x_icon1_k32 0.040, x_con05_k24_tfidf0_z 0.0397, x_con05_k16_tfidf0_z 0.0393, x_icon05_k64_z 0.033 …
- **/workspace/huawei/v5/submission_v5.csv** written 19:32: 529,390 rows, 0 missing preds, min 0.000569, mean 0.06265, max 0.7233, sha256 efef5f3d8c978ef83fc5d5b3b59673e5030a5eac4c1aa291cd51850b1a56cb41
- v5/validate.py: VALID (rc 0). v3/validate_submission.py: VALID, same_order_as_example=True (rc 0)
- Spearman vs v2work/out/submission_v3.csv: 0.97471
- Nothing was uploaded or submitted.

### Status 19:39:32
Step C: runC.sh (C1 check -> build -> screen) is waiting on Grok Bot's v7 screens. runC2.sh is queued: gate vs scr_Astack, then vC confirm, decide_vs.py vC vA, and only if it passes, submission_v5c.csv. submission_v5.csv will not be touched. combo.py gained an optional XF3 dir join (defaults unchanged).

### Status 19:57:29 (stopping; near time limit)
Step C has not started. guard.sh is yielding to Grok Bot's v2work/v7_q2.sh: its v7 screens passed, and it is now running ens.py v7base/v7slow for cuts 25/26/27, then blend_v7 and lt_* jobs (likely 1 h+). Chained and autonomous: runC.sh (PID 357096: ppmi check -> feat_pp build -> scr_C1 25/26), then runC2.sh (PID 412156: gate vs scr_Astack Δ25≥+0.0005 & Δ26≥0 -> vC 3-seed confirm -> decide_vs.py vC vA (≥+0.001 on 25/26/23, weekday not worse) -> only then submission_v5c.csv + validators + Spearman). Outputs: out/runC.out, out/ppmi_check.log, out/runC2.out, out/c1_screen_gate.log, out/decide_vC.log. ppmi.py has not been executed yet, so a runtime bug is possible. C2 (null-importance) not run. C3 (DART) needs Grok Bot's OK; no CatBoost. submission_v5.csv untouched. Nothing uploaded.
