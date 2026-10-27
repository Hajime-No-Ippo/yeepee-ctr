# v10 checkpoint notes (sequence model stacked before LightGBM)
- dev seq: seq10.py (pre-share copy ckpt_seq10_pre_share.py). Full v8 config (6000 steps) -> feat_sq10/ ; 3000-step cap -> feat_sq10c/.
- raw sq_logit AUC  full: C25 .78236 C23 .73516 C26 .79276 | cap3000: C25 .75478 C23 .72531 C26 .75839  (cap undertrains badly)
- ablations (w = post-blend weight; delta vs v8 trim C25 .81321 / C23 .79056 / C26 .82188): see cmp10_results.jsonl
  v10L logit only w0: +.00018 / -.00022 / +.00092 ; v10A all w0: +.00079 / +.00061 / +.00116 ; v10A w0.2: -.00005/-.00156
  v10B (drop raw pdot/pcos/mdot/mcos, drift KS .09-.16) w0: +.00067 / +.00046 / +.00117   <- chosen feature set
  v10C = v10B feats on cap3000 seq: -.00051 / -.00029 / -.00021  -> cap rejected
- runtime: full fits J23..J27 ~350s -> ~900s total (over). Testing shared-fit scheme (run_share.sh, feat_sq10p/q, tag v10P).
- distcheck: logs/distcheck_sq10.txt
- solution builder: make_sol10.py MAXSTEPS -> sol/solution_v10.py
## 16:31 (Europe/Dublin) speed sweep, raw sq_logit AUC on val25 (single J25 fit, 490k rows)
- v8 cfg bs256 lr2e-3 ep3: .78236, 80s | bs512 lr2e-3 ep3: .76809 81s | bs512 3e-3 ep3: .78349 82s | bs1024 4e-3 ep3: .78207 75s
- bs512 ep2: lr3e-3 .76263 55s | lr4e-3 .78170 56s | lr6e-3 .77841 56s  -> batch size doesn't speed up (compute ~ samples*epochs); ep2 saves ~30%
- shared-fit v10P rejected (C25 +.00027, C23 -.00037). Next: v10E = ep2 bs512 lr4e-3 full folds -> feat_sq10e, LightGBM 25/23/26
## 16:50 v10E (ep2 bs512 lr4e-3, ~30% faster fits) rejected: C25 +.00014, C23 +.00015 (rw +.00013), 24-48h -.0006
- New runtime plan: keep exact v10B seq config (v8 seq: 3 ep, bs256, lr2e-3, <=6000 steps) but OVERLAP it: bgseq.sh starts seq10 right after prep_v3
  (needs only v3/data) with SQ_THREADS=3, polars stages (~1.8 idle cores in v8 timings) run alongside; waitseq.sh joins before base.
- sol/solution_v10.py built by make_sol10.py 6000; timed with timed_run_v10.py (copy of v8b harness), work /workspace/huawei/timed_work_v10
## 17:20 overlap run (SQ_THREADS=3 bg + polars stages) KILLED: oversubscription thrash (feat_x 427s vs 56, J25 fit 497s). Rejected.
## 17:30 seq thread scaling (J23 fit alone): 1 thr 69s, 2 thr 61s, 4 thr 54s -> torch barely scales. New plan: seq pinned to core 3 w/ 1 thread,
  overlapped stages pinned to cores 0-2 with 3 threads (make_sol10.py v3; ckpt_make_sol10_v2.py = previous). Timed run -> timed_work_v10b
## 17:29 timed run v10b (make_sol10 v3, seq pinned core 3 / 1 thr, overlap on 0-2): TOTAL 952.7s -> OVER 840.
- anomaly: feat_v2 282s (v8: 55s) and seq f5 tokenize 228s (normal 5-8s) during the same window 17:16-17:20; seq bg ended 17:26 (758s) -> critical path.
- output valid (529,390 unique rowids = eval, p in [0.0009,0.82], no nulls), Spearman vs v8 sub .988; seq feats vs experiment feat_sq10 Spearman .996-.998 (1-thread nondeterminism ~ seed noise).
- saved out/submission_v10_timed952.csv
## 17:35 diag: reran exactly that collision (seq f5 on core 3 + build_folds lite val26/eval on 0-2): bf 33s, seq f5 116s (tok 7s) -> NOT reproducible; slowdown was a transient external contention.
  Expected critical path w/o anomaly ~ 33+~525 seq + base 74 + slow 92 + 10 ~= 730s. Re-timing the SAME solution once (no model change) -> timed_work_v10c.
## 17:49 re-time (same solution, timed_work_v10c): TOTAL 727.9s (< 840 margin, < 900 limit). seq bg fits normal (f3 75 f4 96 f5 119 f6 119 eval 120s), feat_sq_wait 155s = seq is critical path.
- seq feats bit-identical to timed run #1 (deterministic 1-thread); vs experiment feat_sq10 Spearman .997/.998. Final preds Spearman .9978 vs run #1, .988 vs v8.
- valid: 529,390 unique rowids = eval, p in [0.00092,0.823], mean .1087, no null/NaN.
- out/submission_v10.csv sha256 bee2c7d579456f1781155db491c96455eafb09a70da3294d6601b17bf17980ec
- out/submission_v10_candidate.csv.zip sha256 f2318ed451f4310387db244ecd2fc4ecb9518b14b90508dfe9b126d9e74bc861 (plain zip, NOT final)
- gate: package_final.py --auc25 0.81388 --name v10 -> GATE FAIL (0.81388 < 0.83), nothing packaged (expected).
- scores (v10B experiment = shipped feature set): C25 .81388 (+.00067) C23 .79102 (+.00046) C26 .82305 (+.00117) vs v8 trim .81321/.79056/.82188.
- cleanup: removed timed_work_v10c (copied sub out), scr dirs. timed_work_v10b/timed_work_v10 left for Eric's call.
