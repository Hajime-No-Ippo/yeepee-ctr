# Yeepee v5 — implementation brief (Spec Polisher, 2026-10-23 ~14:55 IST)
Polished from Code Buddy's /workspace/huawei/v5/BRIEF_v5.md + AUC Researcher's research/v5_ideas.md.
**Do not run until Eric confirms in Code Buddy's chat.**

## Hard constraints (apply to every step)
- `/workspace/huawei/v2work` and `/workspace/huawei/v3` are READ-ONLY. Code imports from `v3/` (features.py, split.py) are fine; nothing may write there.
- No item, co-click, SVD/idf, KMeans, popularity or global stats from eval rows or eval histories. Item/global side = CLg (train-only clicks, ts < C) and P_rows (train impressions, ts < C). A target row's OWN pre-C history may only feed that user's own vector.
- One heavy job at a time, ≤3 threads (LightGBM `num_threads=3`, `OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3`, `nice -n 19`). Before each heavy job: `pgrep -af 'ens.py|combo.py|screen.py|extra|v3fold|build_folds'`; if Grok Bot's ens.py is running, wait.
- No uploads, no Kaggle/portal calls. Write `submission_v5.csv` only if the decision rule passes. Eric uploads himself.
- Excluded (failed in v4 or owned by Grok Bot): stale/eval-matched folds, per_day, lambdarank, choice/event stacking, seed bagging/wide models, rcov.
- Box: 8 cores, 15 GB RAM, ~76 GB free disk.

## Step 0 — set up /workspace/huawei/v5 (no new features yet)
Copy, don't move:
```
cd /workspace/huawei
cp -a v2work/{v3feat_ne.py,v3fold.py,feats.py,build_folds.py,extra.py,extra2.py,combo.py,cl.py,screen.py,validate.py,write_sub.py} v5/
cp -a v2work/{feat,feat_v3,feat_x,feat_x2} v5/          # ~6.1 GB; only f1..f6, h1..h5, val25, val26, eval are needed
(cd v2work && find . -maxdepth 2 -type f -printf '%p %s %T@\n' | sort) > v5/v2work_manifest_before.txt
```
Path patches (every file above): replace `/workspace/huawei/v2work` with `/workspace/huawei/v5` in `sys.path`, `OUT`, `D`, `P2`, `P3`, `CACHE`, the results jsonl/npy paths in combo.py/screen.py, and model/`OUTP` paths. Keep `/workspace/huawei/v3` imports and `load('/workspace/huawei/v3/data')` (read-only use).
Thread patches: combo.py `PARAMS num_threads=8 → 3`; screen.py `--threads` default 8 → 3.
Add a weekday cutoff, val23 (cutoff Thu 23 Jan, first event/user in [23 Jan, 27 Jan), trains on f1,f2):
- `v3fold.py`, `extra.py`, `extra2.py` SPEC: `'val23': (J20 + 3*DAY, None)`.
- `build_folds.py`: add a `c23` spec `[(J20+3*DAY, 4*DAY, 'val23')]` (h1/h2 already exist).
- combo.py/cl.py `V2_OF['val23']='val23'`; screen.py `efold` → `.get(a.cut, [])` so cut 23 works with `--folds f`.
- Build val23 in all four families: `build_folds.py c23`, `v3fold.py val23`, `extra.py val23`, `extra2.py val23`.

**Acceptance 0**
- `rg -n "v2work" v5/*.py` returns nothing.
- Baseline reproduction in v5 with the shipped config (notime, `XF=1 XF2=1`, f-folds, 400 rounds), seeds 42/43/44: val25 mean within ±0.0005 of 0.7775 (seed spread seen: 0.7772–0.7778). Also record val26 and val23 (all + weekday segment). Save as `v5/out/base.json` — this is THE baseline for the decision rule (same machine, same threads, same seeds).
- 325 features in the base featlist.

## Step A — one extra.py rebuild (ideas 1+2+3)
A0. Leak fix (audit flag, extra.py L51/L55–58). Add a train-only user embedding `Ug = nrm(Xc @ (Vt.T/np.sqrt(S)))` (Xc is already CLg-based). Use `Ug` for `iC = Scg.T @ Ug`, `iN = Sn.T @ Ug`, and the KMeans **fit** (`Un[act] → Ug[act]`). Keep `Un[uu]` for the target-user side and `km.predict(Un)`.
- Acceptance: for the `eval` table, iC/iN/KMeans centres built with `T_ev=EE` vs `T_ev=EE.head(0)` are identical (max |diff| = 0, fixed random_state). Rebuild base feat_x with only this fix → `base_fix` must be within −0.0005 of base on val25 (it becomes the new comparison base if it moves).

A1. `v5/con_sweep.py` (copy of extra.py `build()` lines ~20–38; val25 only; no LightGBM). Pass 1 grid, α=0.5 fixed: k ∈ {16,24,32,48} × weighting ∈ {tfidf, binary no-idf, BM25 (k1=1.2,b=0.75, doc lengths from CLg), log1p(count)·idf} × β ∈ {0, 0.5, 1} (Vn = nrm(Vt.T·S^β)) = 48 configs. Pass 2: α ∈ {0.25, 1} for the top 6 from pass 1.
- Per config log to `v5/out/con_sweep.jsonl`: global AUC and within-event AUC (mean per-event AUC over clicked events) for raw and `_z`, Pearson vs x_con05_k32, runtime.
- Selection: top by within-event AUC, keep 2–4 with |corr| < 0.9 vs x_con05_k32 AND vs each other. Name `x_con{α}_k{k}_{w}{β}` + `_z`/`_rk`.
- Acceptance: the config (k32, tfidf, β=0.5, α=0.5) reproduces x_con05_k32 on val25 with corr ≥ 0.99 (within-event 0.7257 ±0.002).

A2. Item side in extra.py k loop, using Ug: `x_icon05_k32`, `x_icon1_k32`, `x_icon025_k64`, `x_icos_k32` (+`_z`/`_rk`).
A3. Taste confidence (k32 + the best A1 config): `x_con05_k32_norm` (‖nC−0.5nN‖ pre-normalise), `x_con05_k32_dot` (unnormalised), `x_nclk_cl`, `x_nskip_p`, `x_con05_k32_dmax` (raw − event max). No `_z` for counts.
A4. Drop exact duplicate `x_con05_k64` (= sv_k64_con05) from the featlist.
- Rebuild feat_x for f1..f6, val23, val25, val26, eval (one process, sequential; ~10–40 s/table). e/E/d folds not needed.
- Acceptance: no NaN/inf in new columns (counts may be 0); row count and rowid set identical to old feat_x per table; new column count printed.

## Step B — slate context (idea 4), `v5/slate.py` → `v5/feat_sl/NAME.parquet`
Pre-check on val25 first (one k32 SVD on CLg): within-event AUC of `sl_cand_cent_cos`. **Stop Step B if ≤ 0.55** (log the number).
If it passes: `sl_cand_cent_cos`, `sl_disp`, `sl_user_cent`, `sl_cand_top_cos`, `sl_pop_rank` (train-only popularity). Uses only the row's own 10 candidates. Join in combo/cl via a `XSL=1` env flag.
- Acceptance: each event has exactly 10 rows; `sl_disp` constant within event; no stats across events.

## Step C — only if A or B was accepted
C1. `v5/ppmi.py` (idea 5): CLg sorted by (u, ts), ±5 window, weight 1/dist, shifted PPMI (−log 5), randomized SVD k32/k64 → `x_pp_con05_k32`, `x_pp_last3`, `x_pp_hcos` (+`_z`/`_rk`). First: Pearson vs `feat_rc` and seq.py outputs; skip if any |corr| > 0.9.
- Acceptance: assert window counts built from CLg only (eval/target histories excluded).
C2. Null-importance pruning (idea 6): 1 real + 5 shuffled-label screens (25% users, 300 rounds). Drop features with real gain < 90th pct of shuffled gain. Accept only if it improves val25 AND val26.
C3. DART (idea 7): lowest priority; ask Grok Bot before touching ens.py. No CatBoost install without Eric.

## Screening protocol (exact)
1. **Single-feature screen** (Step A1/B pre-check/C1): val25 only, no training; criteria as above.
2. **Cheap model screen**, per candidate feature group vs the current base, run one at a time:
   `screen.py 25 TAG --frac 0.25 --seeds 2 --rounds 300 --threads 3 --folds f` and the same at 26. Base screen uses identical frac/seeds/rounds (`TAG=base_v5`). Candidates add columns only.
   Pass = mean ΔAUC ≥ +0.0005 on val25 AND ≥ 0 on val26. Groups that pass are stacked cumulatively, re-screened once together.
3. **Full confirmation** (only for the final stacked set): combo.py val, full f-folds, 400 rounds, seeds 42/43/44, on val25, val26, val23. Same code path as `out/base.json`.
Log every run to `v5/out/results.jsonl` with: tag, step, cut, seeds, n_feats, AUC at 300/400, seg AUCs (warm, history_only, empty_history, weekday, weekend, stale_lt1d, stale_ge1d), event AUC (max y vs sum p per event, full-model probs only), top-15 gain, runtime.

## Decision rule
Write `/workspace/huawei/v5/submission_v5.csv` only if the final v5 set beats `out/base.json` (same rows, same seeds, mean of 3) by **≥ +0.001 on val25 AND val26 AND val23**, and the val23 weekday segment is not worse than base. Otherwise write no submission and report.
If it passes: `combo.py final N_ITER 3 notime` with N_ITER = the round count from v2work's ne_final_v3 run (check solution_v3.py; 400 unless it says otherwise), f-folds, 3 seeds, `num_threads=3`; `write_sub.py` → `submission_v5.csv`; run `v5/validate.py` and `/workspace/huawei/v3/validate_submission.py` (529,390 rows, header `rowid,predicted_probability`, all in [0,1], unique rowids, no NaN). Report Spearman vs `v2work/out/submission_v3.csv`. Plain probabilities only — no per-event softmax/rank.

## Final checks before reporting
- Re-run the v2work manifest command; diff against `v5/v2work_manifest_before.txt` must be empty.
- `rg -n "v2work" v5/*.py` empty; no file written outside /workspace/huawei/v5.

## Report back (to Spec Polisher and Eric)
One table per step on val23/val25/val26: AUC, Δ vs base, stale buckets, weekday/weekend, event AUC, top-15 features, runtime. Then: decision-rule pass/fail with numbers, and the submission path + validator output if written. State plainly that nothing was uploaded.
