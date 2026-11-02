# dw_blend findings (2026-11-02, Europe/Dublin)

## Headline
- Current best public: S1 = 0.829101 (27th). Target 15th = 0.831467, gap +0.0024.
- S1 recipe: rank(EN) - .109*rank(E0) + .123*[(1-.665)*A + .665*D] - .052*v8regen - .033*v5. The main line is the teammate's EN (0.828856). Our own models add only ~+.00025 on top of it.
- Blending alone cannot close +0.0024. Nothing I found clears the +.0003-at-both-C23-and-C25 bar. No candidate file was produced in out/ and nothing was uploaded.

## Inventory (inventory.csv, 246 rows)
- 236 val-bearing prediction keys scanned from 579 npy files (scan_raw.csv, scan_summary.csv), plus EN, E0 and S1, which have eval predictions only.
- Public scores are known only for a handful of models.
- Only about 25 keys have both C23 and C25 fold predictions. Of those, only the ~10 with eval arrays can go into a submission.
- The transductive flag is heuristic: 12 keys are tagged as B variants.
- The best fold models are all our own v21 family: ctl6/M (C23 .7986 / C25 .8190), big, A, list1, D, z. Rank correlations between them are .985-.9996.
- Older families (v8-v19) have AUC .786-.797 at C23 and are lower by .002-.012.

## Diversity (corr_eval.csv, eval Spearman)
- Ours vs EN: rho ~.956. The own-family members are one signal.
- The lowest-rho components are weak: p33 (rho .91, public ~.805), v5/v8regen, lr9d (rho .936 with M, C25 .8086), xendcg (.934, .8096), tA (.956).
- D_dcn is somewhat diverse (rho .95-.99 depending on variant).

## Local blend search (search.py, pool of 29 members, global rank space, seed-averaged)
Base = .4z+.6M: C23 .79902, C25 .81921.
- Equal rank-average of the 7 cb_blend members: +.00007 (C23) / +.00030 (C25).
- Greedy forward selection, adding lr9d .1, then D .2, then big .1: +.0002 (C23) / +.00039 (C25). This is in-sample selection and misses the bar at C23.
- Signed LR in rank space over all 29 members (negative weights allowed):
  - Fit on C23, test on C25: +.00060.
  - Fit on C25, test on C23: +.00008-.00012.
  - It fails the both-cuts bar. C23 is consistently the cut that gains nothing.
- Level-2 LightGBM stacker (stack.py) on the logits of the 7 members, with and without event-dev features:
  - C23->C25: +.00006 / -.00001.
  - C25->C23: -.00057 / -.00054.
  - User-grouped CV: C23 -.00047, C25 -.00012.
  - It is worse than a plain rank average. The meta-feature runs were still in progress at the time of writing. If they finish, results will be in stack.log / stack_results.json; I expect the same conclusion.
- Not testable locally: logit-average vs rank and per-event vs global normalisation (the old components have no c23 logits at the same scale; rank/global was used throughout, with no difference visible on the base). EN/E0/p33 have no fold predictions, so their weights can only come from public probes.
- v21cat / v21graph eval arrays had not landed (v21fit/out has no cat_eval.npy/graph_eval.npy) when I wrote this. Run `v21fit/reweight.py` on them when they do.

## How much can blending close? (honest)
- The binormal break-even from Grok Bot's analysis says an added model needs roughly (AUC .8295, rho .97), (.827, .95) or (.8225, .92) to give +.0022.
- Ours are (.819, .956), so a gain of +.0001 to +.0003 at most (the local OOF blends above agree, and the C23 side is lower).
- Blending can close ~10% of the gap, i.e. S1 -> ~.8293-.8294. It will not reach 15th (.8315). At best it gets to about 22nd-25th.

## Which new models are worth training
Need AUC ≥ ~.825 public with rho ≤ .95 to EN, i.e. a genuinely different signal and not another seed of ctl6.
1. A model class that uses information the EN/ctl6 line does not: transductive per-user eval-history sequence models (full eval-fold history, future-aware), and item co-click graph / GNN embeddings (Grok Bot's graph run).
2. A differently-tokenised sequence model (event-level transformer on raw ad/slate IDs, not engineered features), as v21seq-style models share the feature base with M.
3. GBDT with target encodings built on eval-row statistics (now allowed): the v12x xgb/f_all are ~.003 weaker locally but rho .977 with M, a bit more diverse than the NN variants.
4. A reproduction of whatever ErBrown's EN does differently (public .8289 alone vs our .819 alone). That gap of .010 is the actual problem, not the blend weights.

## Probe plan (max 3 submissions; per Eric's rule, only if cat/graph or a new model lands)
Nothing to probe from blending alone. If something lands:
1. S1 + 0.1-weighted new component (if reweight.py gives P>0>.7): expected +.0000 to +.0003.
2. p33 at small weight (.03-.05) on S1: expected -.0001 to +.0002 (public ~.805, unsupported).
3. EN-only (S1 with our models removed) is already known: .828856, so skip.

## Files
scan.py, scan_raw.csv, scan_summary.csv, corr_eval.py/.csv, stack.py/.log, pool.py (pool_c23/25.npz), search.py/.log, inv.py, inventory.csv.
