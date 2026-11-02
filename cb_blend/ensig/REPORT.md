# ensig — which existing signals could move S1 (0.829101) toward 0.8297?  (2026-11-02, Europe/Dublin)
Diagnostic only. Read-only on all other folders, CPU taskset 0-3 / 4 threads, no features or training on eval rows, no uploads/submissions/commits, no candidate files.
Files here: `inventory.csv` (source/order/alignment), `results.csv` (all numbers), `ensig.py` (loaders+registry), `run1..5.py, mk.py` (scripts), `scan_cols*` (column convention).

## 1. Inventory / alignment
* 33 members with an eval-side array (529,390 rows). All brought to `merged.parquet` rowid order (== eval.parquet == evaluation.csv; v21seq/xgb8/graph arrays re-joined by rowid; `tab_eval.npz` order is different and is NOT used). Multi-seed members = seed mean in logit space (log-probs converted), final (last) column, then global rank/N.
* Alignment check = rho vs EN in merged order is .85–.985 for every member, while the same vector in a shuffled order gives ≈ −0.002 (see `inventory.csv`). S1 rebuilt from `pS1.npy` + mkS formula equals the S1 zip (Spearman 1.0, max abs diff 0).
* Local fold arrays (C23/C25/C26, `v12/feat_v3/val*.parquet` order) exist for ~30 of them; row counts asserted per file.
* **Members with NO eval array anywhere** (fold predictions only): v21cat `cat_numte`, v20 `ref484`, v21gfeat `gf_num_s3` / `gf_numg_s3`, other v21dcn variants. EN/E0/p33/v8regen/v5/EBDISTILL/O have eval arrays but no fold predictions.
* Data-rule flags: coBp = variant B (transductive); EBDISTILL was trained on EN's outputs (not variant-A clean); EN/E0/O/p33 are teammate/ErBrown arrays.

## 2. Eval-side orthogonality (rank space, global)
rho vs EN, rho vs S1, and residual sd of rank(member) after OLS on [1, rank(EN), rank(S1)] (uniform-rank sd = .289; `resid_ratio` = resid sd / member sd = sqrt(1−R²)).
Members already inside S1 (E0, A_all, D_dcn, v8regen, v5) are listed for reference only.

| member | rho EN | rho S1 | resid sd | public/local AUC proxy |
|---|---|---|---|---|
| v2base / v2slow / estrc / wide (v2work, "ne cleared" LGB) | .85 | .84 | .143 | .778 (C25) |
| v5 (in S1, −.033) | .871 | .863 | .129 | – |
| graph7s | .905 | .901 | .118 | .795 |
| p33 | .909 | .909 | .120 | .805 pub |
| v6base | .910 | .907 | .118 | .802 |
| lr (v9d) | .935 | .933 | .101 | .809 |
| ctl_b (v9d) | .945 | .943 | .094 | .811 |
| v7base | .946 | .944 | .093 | .811 |
| xgb(v12x) / dwbig / dw_s / dwmlp / mlp2 / sCbig | .948–.953 | .949–.953 | .088–.091 | .815–.8165 |
| z_lgb / z_mlp / coAp / coAmlp | .951–.952 | .953–.955 | .084–.087 | .817–.8175 |
| ctl6(M) / big / list / O | .954–.956 | .958–.959 | .078–.079 | .819 |
| A_all (in S1) / D_dcn (in S1) | .956 / .950 | .960 / .955 | .079 | .8197 / .8189 |
| EBDISTILL | .965 | .967 | .073 | ≈.8197 assumed |
| E0 (in S1) | .985 | .981 | .044 | .8246 |
Nothing with AUC ≥ .815 is below rho .935 to S1. The only low-rho members (v2 family, v5, graph, p33, v6) are weak (.778–.805). Full numbers: `results.csv`.

## 3. Local held-out gain (no EN here — see caveat)
Base = (1−.13)·mean(rank z,M,big,D) + .13·rank(graph0312) (C23 .79979, C25 .82025); C26 base uses M,big only (.82865). Gain of adding X at weight w (grid .02–.3); "HO" = w picked on the *other* cut, gain in e-4 AUC.

| member | rho vs M (C25) | best gain e-4 C23 / C25 / C26 | held-out e-4 (C23 w from C25 / C25 w from C23) |
|---|---|---|---|
| gf_numg_s3 (v21gfeat, graph feats; NO eval array) | .979 | 1.87 / 1.00 / 2.67 | **+1.76 / +0.94** |
| lr (v9d) | .935 | .31 / .34 / .47 | +.28 / +.33 |
| v2base, v2slow, estrc, wide (C25→C26 only; no C23) | .81–.82 | – / .28–.42 / .48–.58 | +.42–.48 on C26 |
| cat_numte (v21cat; no eval array) | .980 | .58 / .05 / .24 | +.33 / −.30 |
| gf_num_s3 (control LGB, no eval array) | .980 | 1.00 / .03 / .27 | +.26 / −.61 |
| v6base / v7base (C25→C26) | .91 / .95 | – / .27,.09 / .13,.07 | +.11 / +.07 |
| dwbig, xgb, dw_s, coAp, z_lgb, ref484, graph7s (7-seed, already in base) | .97–.99 | ≤.7 (C26 only) | ≈0 or negative |
| ctl6, big, list, dwmlp, mlp2, sCbig, z_mlp, coAmlp | .97–1.0 | 0 | 0 |
Greedy forward selection (nonneg, w in {.03,.06,.1,.15,.2}, fit on one cut, scored on the other): C23→C25 reaches +2.0e-4 on C23 but only **+0.7–1.1e-4 held-out on C25**; C25→C23 +1.2e-4 fit, **+1.8e-4 on C23**. Picks: gf_numg_s3, lr, xgb, gf_num_s3. (C23 is the weaker, noisier cut; cross-cut weights are unstable, same pattern as Data Wrangler's findings.)

**Caveat that dominates everything:** these are gains over an eq4(+graph) base *without EN*. The graph is the calibration case: local +7e-4 over eq4, but public gain over S1 (which contains EN) was +1.2e-5 for G2 (−1.46e-4 for G1). rho(graph,EN)=.905 on eval; EN already carries most of it. Transfer ratio local→public for an EN-explained signal ≈ 2–10%.

## 4. Predicted public gain over S1 (binormal model, `run3.py`/`run5.py`)
d'=√2Φ⁻¹(AUC), combined d'=Σw·d/√(wᵀRw), R = eval rank correlations. AUC proxy = public where known, else local C25.
* **Calibration is only partly good:** Q2 predicted +1.68e-4 vs actual +1.60e-4 (good); R1 +5.9e-4 vs +2.0e-4, S1 +8.7e-4 vs +2.45e-4 (model over-predicts, since v8regen/v5 AUCs are guesses and their weights are negative); G2 −4.4e-4 vs +0.1e-4 and G1 +3.3e-4 vs −1.5e-4 (sign wrong: graph is a borderline case, model says d_G/d_S1=.857 < rho=.90, actual says slightly positive). Use the model for the *shape of the break-even frontier*, not for point predictions.
* First-order criterion: adding X helps only if AUC(X) is above the curve AUC ≥ Φ(rho·d_S1/√2): rho .85→.791, .90→.804, .93→.812, .95→.817, .97→.822.
* Needed AUC for **+2e-4 / +6e-4** over S1 at optimal weight: rho .85: .796 / .8005; .90: .8085 / .812; .93: .8155 / .818; .95: .820 / .822; .97: .824 / .826; .98: .826 / .828.
* Every existing member with a free eval array is predicted ≈0 (≤ +0.2e-4) over S1 at best (D, A, big, M ≈ +0.0–0.15e-4; already in S1). v5 shows +2.4e-4 only because its AUC is an assumption and it is already in S1 with negative weight.

## 5. Shortlist (what could add ≥ +2e-4 on top of EN/S1)
Ranked by (low rho to EN/S1) × (local held-out gain) × (eval array exists):
1. **gf_numg_s3 (LGB + 3-seed graph features)**: best held-out local gain (+0.9–1.8e-4 over eq4+graph, +2.7e-4 on C26) but it is graph-derived, rho_M .979, and the graph member itself got only +1.2e-5 on S1. Expected public gain ≈ +1e-5…+5e-5. No eval array (needs eval graph k=7 features + inference).
2. **lr (v9d logistic)**: rho EN .935 / S1 .933, AUC .809, stable held-out +.3e-4 on both cuts and C26. Binormal says needs AUC .8155 at rho .93 for +2e-4; it has .809 → ≈0 to +0.3e-4 public. Cheap and safe to try as a tiny-weight add.
3. **v2work family (v2base/slow/estrc/wide)**: lowest rho to EN (.85) but AUC .778; locally +.4e-4 (C25/C26). v5 (same era) is already in S1 at −.033; expect ≤ +0.3e-4, and probably already the thing S1's v5 term absorbs.
4. v6base/p33 (rho .91): weak AUC (.80), local +.1–.27e-4; p33 has no fold preds.
5. Everything else (ctl6/big/list/z/dw*/co*/xgb/mlp2/sCbig/EBDISTILL) is rho ≥ .95 and AUC ≤ .8197 — below the break-even curve or on it → ≈ 0.
**None of these is plausibly +2e-4 over S1 individually; the ensemble of all of them: realistic +0.3e-4…+1e-4 public, unverifiable locally** (no EN fold preds; signed 29-member LR in Data Wrangler fit C23→C25 +6e-4 but C25→C23 +1e-4).

## 6. Missing / untested families (no eval array, or never built) — where a +6e-4 would have to come from
* Session/sequence over *within-slate event time* (v21seq big already covers history DIN; no pure session model).
* Text/title/content embeddings (never built in variant A? none found in the eval arrays).
* Impression-SVD taste family (dw_signals: +0.0012 local over v11 via residual boosting) — probably already inside v21seq/ctl6 features (rho .95 to EN anyway); worth confirming it is in A.
* Cross features / target-encoded cats (v21cat numte: local +.5e-4 C23, ~0 C25; no eval array).
* Cold-start specific models (no-history rows are only .8–1.3% of val; segment AUC .52–.55 → cannot move global AUC).
* Event-level propensity ("any-click" AUC .67; headroom sim says +0.003 needs event AUC .709; every attempt ≤0).
* EN's own source: EN has no fold predictions, so any EN-like model (what ErBrown did) cannot be reproduced locally; ours reaches .819 where EN is .8289 public — the gap (+.0098) is the real lever, and no ensemble of .819-class members closes it (v21fit/Data Wrangler agree).

## 7. Honest odds of reaching 0.8297 (+6e-4 over S1)
* From re-blending existing members (including the above shortlist, no new training): **≈ 2–3 %** (the diagnostics put the ceiling near +1e-4; each member is on/below the break-even curve).
* With a new, differently-structured model (AUC ≥ .822 at rho ≤ .95 to S1, or ≥ .826 at .97): that would itself be a major jump (ours are .819); **≈ 10 %** within the remaining time.
* gfeat 7-seed rerun + eval inference + tiny-weight lr/v2 adds as a package: expected **+0.3e-4 … +1e-4** public, i.e. 0.8292, **≈ 5 %** of hitting 0.8297.
