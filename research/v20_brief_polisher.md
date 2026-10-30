# v20 brief: fewer features, overfitting guards (Spec Polisher pass)
Origin: Eric via Prompt Refiner, 30 Oct 2026. Code Buddy has already started; this refines that work. The v20fast runtime spec (research/runtime_cut_spec.md) is a separate track. Re-use the per-tier fit times from here there.

## Hard rules
- Work only in `/workspace/huawei/v20/`. Treat v17cooc, v19, v18classic, v12 and v2work as read-only. v19 jobs may still be running, so don't kill or overwrite them.
- Variant **A only** (train-only): feat_Ap + feat_NAs/NAp. B uses eval-row histories (transductive) and is excluded.
- One heavy job at a time, `--threads 4`, under `taskset -c 0-3`.
- Adversarial validation (AV) is **diagnostic only**. Any AV-based drop list needs Eric's approval before it reaches a submitted tier. Ask through Prompt Refiner.
- Pick by local time-fold AUC only. Never pick on the leaderboard: the v15 LB numbers are context, not a target.
- Final score = blend in [0,1]. Rank averaging is allowed only as a **global** rank / N. Never per-event rank or softmax.
- No uploads.

## Stack hints
- Trainer: copy `v17cooc/xlgb17.py`, `cl.py` and `ens17.py` into v20 (or v19's copies). xlgb17 takes `cut tag --seed --rounds --save --params JSON --threads --lr <feature dirs>`. Add `--feats <json list>` to restrict columns.
- Reference (484 features): the v19 `nmAs` config, `--lr feat_lr2,feat_tt,feat_pp,feat_sqC,feat_dw,feat_Ap,feat_NAs`, with dwbig params and 1000 rounds. Logged C25 is .81751.
- Time folds: use the existing cuts **C23, C25, C26** (val23/val25/val26 tables; scored on each user's first event after the cutoff). C23 is the weekday/long-gap fold. If any v19 feature dir lacks val26, build it with the same v19 script into v20 (no edits to v19). Don't invent a C24 unless all its tables can be built leak-free.
- Gain files: `v19/ens/nm*_c2{3,5}_s0_gain.json` are seed 0 only. Generate more seeds in step 2.

## Steps (each ends with an acceptance check)
0. **Setup and reference.** Copy the scripts, then rerun the 484 reference with seeds 0, 1, 2 on C23, C25 and C26 → `out/ref.json`.
   *Check:* the C25 seed-0 result is within ±0.0001 of .81751, and the seed spread is logged.
1. **Cheap prefilter (no training).** On a 300k-row sample of the C25 training fold:
   - drop constant or near-constant columns (more than 99.9% of rows share one value);
   - cluster by |Spearman| > 0.98 and keep the member with the highest mean gain.
   → `out/redundancy.json` (clusters, kept, dropped).
   *Check:* every dropped column names the column it was kept in favour of.
2. **Stable importance.** For C23 and C25, seeds 0–2, fit on the prefiltered set with 500 rounds. Record normalized gain (gain / total) and average over the 6 runs. Then compute **permutation importance** on the val rows for the top 300 by gain: shuffle the column globally, take 3 repeats, measure the AUC drop. This is predict-only, so it's cheap.
   → `out/importance.csv` with these columns:
   - feature
   - gain_mean, gain_cv
   - perm_drop_c23, perm_drop_c25
   - rank_c23, rank_c25

   *Check:* the 6 runs finish, and permutation drops are reported with their standard deviation.
3. **Stability filter.** Flag a feature as unstable if gain_cv > 1.0 **or** it ranks in the top 200 at one cut and below 400 at the other. Unstable features are dropped from the tiers, but listed. Run AV in parallel (train-fold rows vs eval rows, LGB with 200 rounds, 3-fold). Flag features with AV gain share > 2% or a single-feature AV AUC > 0.6 → `out/av_flags.json`. **Don't drop these, only flag them.**
   *Check:* the AV run reports its overall AUC and the top 20 features.
4. **Feature-count curve (quick).** Rank the kept features by permutation drop averaged over C23 and C25, breaking ties by gain. Fit C25, seed 0, 500 rounds at N ∈ {25, 50, 75, 100, 150, 200, 250, 350, all kept}.
   *Check:* plot or table of AUC vs N. Choose the 3–4 tiers at and just past where the curve flattens, roughly 50/100/150/250 unless the data says otherwise.
5. **Regularized params (per chosen tier, small grid).**
   - num_leaves {63, 127}
   - min_data_in_leaf {200, 500}
   - feature_fraction 0.6, bagging_fraction 0.8, bagging_freq 1
   - lambda_l1 {0, 1}, lambda_l2 {1, 10}

   Pick the number of rounds by early stopping on **C23**, then apply that fixed count to C25 and C26 so the scored folds never set their own stopping point. Screen at C25 with 1 seed first and confirm the top 2 configs on all folds.
   *Check:* the chosen config is no worse than the dwbig params on mean AUC over the 3 folds.
6. **Confirm tiers.** For each tier and the AV-dropped variant of the best tier, run the best params on C23, C25 and C26 with seeds 0–2. → `out/tiers.csv` with these columns:
   - tier, n_feat, params
   - auc_c23, auc_c25, auc_c26
   - mean, std over folds, seed_std
   - fit_s per fold (4 threads)

   Put the 484 reference in the first row.
   *Check:* the table is complete.
7. **Blend (optional).** Use at most 3 diverse members: best-tier LGB, dw_s-style small LGB, and one MLP on the same tier. Fit the weights on C23 out-of-fold predictions only, then score them on C25 and C26. Use the logit mean or the global rank mean.
   *Check:* the blend beats the best single member on the mean of the 3 folds, or else drop it.
8. **Submission.** Do the cut-27 fit of the winning tier (all 3 seeds) → `out/submission_v20_<tier>.csv` and `.zip` with its sha256, and `out/feats_<tier>.json` for every tier. Validate:
   - header is `rowid,predicted_probability`;
   - 529,390 rows;
   - rowids equal `example_submission.csv`, in the same order;
   - values finite and in [0,1].

   Also report the Spearman correlation with the v19/v17cooc A submission.
   *Check:* the validator passes, and the file is not uploaded.

## Decision rule (done when)
A tier qualifies if both hold:
- its mean AUC over C23/C25/C26 is ≥ (484-reference mean − 0.0005);
- its fold std is ≤ the reference's.

Pick the smallest qualifying tier. A difference below about 2× the seed std (~±0.0001) counts as a tie. If no tier qualifies, report the curve and stop: don't submit.

## Report to Eric
Send him `out/tiers.csv`, the chosen tier, its feature list path, and the AV flags awaiting his approval. Include the fit time per tier on 4 threads, which feeds the 15-min runtime track.
