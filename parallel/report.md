# v2work vs v3: leakage audit and like-for-like comparison (parallel check)
Started 23 Oct 2026 12:35 (Europe/Dublin). v2work and v3 were only read. Everything here lives in /workspace/huawei/parallel/.
The source snapshot comes from v2work/out/solution_v3_src.zip (06:24) and is unzipped in parallel/src_snapshot/.
Current v2work code differs from it only by the added `d{k}` fold specs (12:27 edits), so the snapshot is what produced submission_v3.

## Status log
- 12:45 Audit of the code path done (see below). Like-for-like scoring done (score_val.json). Submission check done (subcheck.json).

## 1. Leakage audit of the v2work "ne" (no-eval) path
Citations point to parallel/src_snapshot/v2work/* (identical to what produced out/submission_v3.csv).
C is the cutoff. "Own history" is the history list that is a given input field on every row, including eval rows.

**a. Are training rows only events before the cutoff? YES.**
- v3fold.py L11–12 SPEC: f_k = first event per user in [J20+k·d, +1d).
- In val mode, combo.py L44 trains on f1..f4 only (cut25). All target ts < Jan25.
- Val25 is the first event per user at or after Jan25 (W=None), the same as our v3/split.
- Final (combo.py L68, FOLDS unset in solution_v3.py) uses f1..f6, whose targets fall in Jan21..Jan26 (<Jan27).
- build_folds.py L9–12: val25 is (Jan25, 2d), and h1..h4 are 1-day windows.

**b. Are label-derived features computed only from labels before the cutoff? YES.**
- v3feat_ne.py L109–110: P_ev and P_rows are filtered to ts<C. All CTR, impression and click stats come from these.
- feats.py L40–41: same filter.
- extra.py L22–24 and extra2.py L29–31 assert `CL.ts.max()<C and P_rows.ts.max()<C`.
- Cluster CTR (extra.py L60) and user state (extra2.py L53–59) come from P_rows and P_ev.

**c. Does any feature use the unlabelled eval rows? Only their own history, and only for user-level features.**
- Item, global, co-occurrence and SVD stats come from CLg = click_log(..., T_ev.head(0)), which is train only:
  - v3feat_ne.py L113–114 (CL vs CLg)
  - L128 (item ip)
  - L173 (cooc)
  - L195 (SVD from CLg)
  - feats.py L54–55 poolg → L65, L87, L96
  - extra.py L31 (idf from CLg)
  - extra2.py L35–39 (SVD)
- User-level stats use each row's own pre-C history:
  - v3feat_ne.py L209–212 and L222 (user fold-in onto train-only item factors)
  - feats.py L49–50 and L117–122
  - extra.py L36
  - extra2.py L41
- This is standard: every eval row carries its own history. It doesn't pool across eval rows. No eval labels exist or are used.
- Two minor transductive touches, label-free with pre-C clicks only:
  - extra.py L51 builds item-side vectors iC = Scg.T @ Un. Scg is train-only clicks, but Un includes the target users' folded-in vectors.
  - extra.py L55–57 fits KMeans on Un over active users, which can include target users.
  - Neither uses labels or post-C data. Expected effect is small.
- write_sub.py reads evaluation.csv only for rowid order and validation.

**d. Is the validation set the same as ours? YES.**
- val25: 439,640 rows, rowid sets identical, 0 label mismatches.
- val26: 351,380 rows, identical (score_val.json).

**Does submission_v3.csv come from the ne path? YES.**
- It equals v2work/eval_pred_v3.parquet (max abs diff 5e-9) and differs from the old eval_pred.parquet by up to 0.34.
- eval_pred_v3 was written by `combo.py final 400 3 notime` with XF=1 XF2=1 (ne_final_v3.log: 2,105,880 rows = f1..f6, 325 feats, 3 seeds).
- All input tables (feat_v3 at 04:06–04:24, feat at 04:06–04:12, feat_x at 04:06–04:15, feat_x2 at 04:16–04:27) were written by the ne_* runs after the no-eval edits (feats.py and v3feat_ne.py at 04:03).

**FLAG: our own v3 is less strict than v2work here.**
- v3/features.py builds item stats (L127), co-occurrence (L172) and the SVD matrix (L194) from CL, which includes every target row's own history.
- For eval predictions this means our submission_v2.csv pools all eval rows' (pre-cutoff, label-free) histories into item-level stats.
- That's the same pattern Grok Bot labelled a leak and superseded in its old v2.
- It's label-free and pre-Jan27, so it's not label leakage. But it is transductive use of the eval rows. Whether that's allowed is Eric's call.

## 2. Like-for-like scoring (our metric: global ROC AUC, v4/common.report), identical rows
v3 preds come from re-predicting the saved models v3/out/full25.txt and full26_lr05.txt in parquet row order.
The saved v3 *_valpred.npy files have an unrecoverable row order because of an unstable sort in v3/train.py. The re-prediction reproduces 0.75554 and 0.77444.

| set | model | AUC | eval-weighted | stale ≤1d | stale 1–2d | event AUC* | Spearman vs v3 |
|---|---|---|---|---|---|---|---|
| val25 (439,640) | **v3 (ours)** | 0.75554 | 0.75554 | 0.76059 | 0.73424 | 0.64894 | – |
| val25 | **v2work ne notime 325f** (combo_val25_notime.npy, e-fold run, 300 rounds) | **0.77749** | 0.77748 | 0.78199 | 0.75830 | 0.65403 | 0.899 |
| val25 | v2work notime_x2 (pre-ne config, 04:26) | 0.77536 | 0.77535 | 0.77968 | 0.75708 | 0.65328 | 0.899 |
| val25 | v2work screen, base, 25%-user train, 1 seed | 0.77174 | 0.77173 | 0.77642 | 0.75176 | 0.59153* | 0.888 |
| val26 (351,380) | **v3 (ours)** | 0.77444 | 0.77444 | 0.77444 | – | 0.65666 | – |
| val26 | v2work screen, base, 25%-user train | **0.78921** | 0.78921 | 0.78921 | – | 0.59846* | 0.912 |

*Event AUC = AUC of max(y) vs sum(p) per event. The screen preds are raw scores, so their sum isn't comparable; ignore those two event numbers.

Notes:
- The final config (f-folds, 400 rounds) scored 0.77748 in ne_combo25.log, but its preds were overwritten by the e-fold run (0.77749 at 300). The two are equivalent proxies.
- There are no full-data v2work val26 preds. The 25% screen alone beats our full v3 by +0.015.
- The gain is uniform across staleness buckets: +0.021 at ≤1d and +0.024 at 1–2d. It isn't concentrated where leakage would show.

**Final submissions:** Spearman(v2work/out/submission_v3.csv, /workspace/huawei/submission_v2.csv) = **0.918** (Pearson 0.929) over 529,390 rows (all joined). Mean p is 0.0625 vs 0.0616.

## 3. Reproduction and lag-1d leak test (parallel/lag/, one job at a time, 3 threads, seed 42, deterministic)
**Reproduction.**
- Retrained v2work's exact final config (snapshot combo.py PARAMS, notime, XF=1 XF2=1, f1..f4, 400 rounds) on v2work's on-disk tables, read only.
- val25 AUC was 0.77684, 0.77724 and **0.77748** at 200, 300 and 400 rounds. That's identical to ne_combo25.log, so 0.7775 is real and reproducible.
- I also rebuilt the val25 tables from the snapshot code (lag/repro_cmp2.json):
  - feat_x2 is exact.
  - feat (v2) has 31 of 122 cols with tiny diffs (corr ≥0.9997).
  - feat_v3 has 60 of 169 co-occurrence/rank cols differing on ≤2.5% of rows (corr ≥0.989).
  - feat_x has clctr cols differing on ≤5% of rows (corr ≥0.996).
  - These diffs are consistent with thread-dependent randomized SVD/KMeans and tie-breaking. They don't point to a different (eval-pooled) build.

**Lag-1d test.**
- Same target rows (val25 = first event per user at or after Jan25, labels unchanged). All four feature builders were rerun with the cutoff frozen at **Jan24**, so no stat or label from [Jan24, Jan25) is used.
- Predicted with the same model (lag/lagtest.py):

| val25 features | AUC | stale ≤1d | stale 1–2d | event AUC |
|---|---|---|---|---|
| as built (C=Jan25) | 0.77748 | 0.78184 | 0.75895 | 0.65358 |
| lag-1d (C=Jan24) | 0.76470 | 0.76946 | 0.74430 | 0.64837 |

- The drop is −0.0128, gradual, with no cliff. It's smaller than v3's own lag-1d drop (0.7556→0.7391, −0.0165; v3/out/leakcheck.jsonl).
- Even with 1-day-stale features, v2work (0.7647) beats our v3 with fresh features (0.7555).
- A target-day leak would collapse towards ~0.65–0.70 when the target day is removed. That doesn't happen.

## 4. Verdict
**v2work's ne pipeline is genuinely better than our v3.**
- val25: +0.022 AUC (0.7775 vs 0.7555) on identical rows and labels.
- val26 screen: +0.015, even with a 25% train sample.
- The gain holds in every staleness bucket and survives the lag-1d test.

The audit found no label leakage:
- Training targets are before C.
- All label stats come from ts<C, with asserts.
- Item and global stats are train only.
- Eval rows contribute only their own given pre-cutoff history to user-level features.

Grok Bot's claims all check out:
- The ne variant files.
- 0.7775 from ne_combo25 and ne_combo25_eft.
- submission_v3.csv built by the 06:30 combo final run (it equals eval_pred_v3.parquet).
- The old v2work submission_v2 is superseded.

Correlation with our submission: Spearman 0.918.

**Flags for Eric:**
1. **v2work/out/submission_v3.csv appears to have been uploaded already.** v2work/out/submission_log.csv has a row "submission_v3.csv.zip (public score) … 0.7775, 0.788757", so public AUC was 0.7888. This parallel check didn't upload or submit anything.
2. Our v3 (and so /workspace/huawei/submission_v2.csv) builds item, co-occurrence and SVD stats from all target and eval rows' histories (v3/features.py L127/L172/L194). That's label-free but transductive, and stricter rules would call it eval-row use. v2work removed this.
3. Minor: v2work's extra.py L51 and L55–57 still fold target users' (pre-C, label-free) vectors into item-side aggregates and KMeans.

**Final CSV:** it exists. v2work/out/submission_v3.csv: sha256 466f3d64…, 529,390 rows, written 06:37, plus solution_v3.py and src zip.
- Rebuilding from scratch at ≤3 threads would take about 40–50 min: feature tables ~25 min, then final fit on 2.1M rows × 3 seeds × 400 rounds ~12–20 min (the 1.4M-row fit here took ~3 min).
- If all tables are reused, only the final fit is needed: ~15–20 min. I didn't build one.
