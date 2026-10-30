# v20 PLAN (fewer features + anti-overfit). Variant A only. No uploads. Threads: V20_THREADS (default 4), cores V20_CORES (default 0-3).
Reference = v19 nmAs: feat_lr2,tt,pp,sqC,dw + feat_Ap (coA, v17cooc) + feat_NAs (v19), 484 cols, dwbig params (lr .015, 127 leaves, ff .5, l2 30, md 200), 1000 rounds. Logged C25 .81751, C23 .79679.
Folds: C23 (train f1-f2, val23), C25 (f1-f4, val25), C26 (f2-f5, val26); cut 24 not used (no val24 table). C27 = f3-f6 -> eval.
Run:  cd /workspace/huawei/v20; nohup ./chain.sh > logs/chain.out 2>&1 &     (resumable via results.jsonl; each stage waits on other agents' heavy jobs)
Stages (repaired 10-30): [prefilter if out/redundancy.json missing] -> preflight -> s0_ref -> s2_importance (prefilter kept set; gain x6 runs; permutation top300) -> s4_curve -> s5_grid -> s6_tiers -> decide.py -> s7_avdrop (Eric-approved A/B ONLY: top-5 impression feats dropped on decided tier + ref484; NOT adopted) -> s8_submit (undropped decided tier). s3_av is NOT rerun; av_flagged.json (0431) is authoritative, backup out/av_flagged_0431.json.
Standalone: prefilter.py (done), gain_rank.py (done).
Decision: tier mean >= ref mean - 0.0005 and fold std <= ref std (+2e-4 tie tol); smallest wins; else report curve, no submission.
