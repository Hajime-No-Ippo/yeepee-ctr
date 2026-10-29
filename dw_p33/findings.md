# dw_p33 findings (Data Wrangler, 2026-10-29)

## (0) What are the recovered p33 ranks?
- `teammate/work33/p33_recovered_ranks.parquet` is the SOLO "other model" (weight 0.3 in `submission_p33_blend_w70.zip`; recipe reproduces to 5e-9 using `teammate/work/v8.parquet`). The teammate's "0.75 v8 + 0.25 own" text does NOT match the w70 file (0.7 v8 / 0.3 other).
- v8 content left in the ranks: ~0 (bounds 0.12 popularity-type .. 0.25 cold-slope). Solo estimate = the recovered ranks; sensitivity-adjusted versions in `p33_solo_est.parquet` (p33_solo_rank/_pct, p33_solo_adj_a012_*).
- Spearman vs v12 on eval: 0.929 (0.912 if a=0.12 v8 removed, 0.885 if a=0.25).

## (1) Pipeline reconstruction and v12 coverage
| p33 piece | v12 has it? | verdict |
|---|---|---|
| item x item history co-occurrence -> PPMI -> SVD k=12-16 | `lr` family (SVD of click matrix, popularity dropped); `lr12_cos` AUC .7667 | same subspace, but the PPMI-centroid cosine `pp12_cos` is better alone: .7791 (+.012) |
| k=12 vs 16 vs 8 | lr8 .7395 / lr12 .7667 / lr16 .7525 | k=12 is the optimum in both (pp8 .719 / pp12 .779 / pp16 .759) |
| recency-weighted user vector | none useful | no gain: exp tau=3 .7589, tau=10 .7773, pow .7625 all <= uniform .7791 (history order carries ~no info; last3 .7335, max5 .714) |
| user.item dot / cosine | lr*_cos, x_pp_* | covered; pp12_cos adds +0.0002 (C25), +0.0006 (C23) |
| item and user coordinates | item/user vectors partially in feat_lr2 | user coords alone ~noise (+0.0001 / +0.0001), item coords: no standalone value |
| **coordinate products** (user_d x item_d, d=1..12) | only `tm_prod` taste products (+.00019 C25 / +.00028 C23) | **the winner**: 12 per-dim products +0.00056 / +0.00098 |
| position decay | no | nothing (rec group at noise) |
| multi time-point fits (W45/W6/We) | no | `pp12_w3_*` (small window) weak, +0.00003 C25. Not worth it |

## (2) Low-rank claim (training data only)
- PPMI (W=20, 1/d) spectrum: clear gap after 1 + 12 components (popularity + 12); raw user x item click matrix spectrum is noise-like; impression matrix elbow ~10. See `spectra.png`.
- k probe on C25, AUC of user-centroid . item dot: k=8 .719, k=12 .757-.779 (peak), k=16 .759, k=64 ~.69, k=256 .670. Reproduces teammate's .7551 vs .672. Conclusion: the data have a ~12-dim latent structure; k=12 is the right size (teammate said 16, 12 is better in our hands).

## (3) Residual harness (v12 logits as init_score, user-grouped 5-fold, event bootstrap 95% CI; delta vs control)
| group | n | C25 | C23 |
|---|---|---|---|
| noise control | 1 | -0.00006 [-.00012,-.00001] | -0.00002 [-.00008,.00005] |
| cos1 (pp12_cos,_z,_dm) | 3 | +0.00011 [-.00002,.00021] | |
| cos (all cosine variants, 13) | 13 | +0.00022 [.00007,.00038] | +0.00058 [.00042,.00078] |
| rec (recency/position) | 6 | -0.00003 [-.00015,.0001] | +0.00006 [-.0001,.0002] |
| **prod (12 coord products)** | 12 | **+0.00056 [.00035,.00076]** | **+0.00098 [.00072,.00119]** |
| prod+item coords | 24 | +0.00042 [.0002,.00062] | |
| user coords | 12 | +0.00010 [0,.0002] | +0.00013 [0,.00024] |
| multi (small window) | 2 | +0.00003 [-.00007,.00013] | |
| all | 57 | +0.00056 [.00035,.00077] | **+0.00128 [.00103,.0015]** |
Ranking: prod > cos > user > rec ~ multi ~ noise. Beyond the noise floor (+-0.0001) only prod and cos. On the C25 cut cos adds nothing over prod ("all" = prod); C23 prefers all.
Expected gain on v12: roughly +0.0006 .. +0.0013 AUC on the residual head (C25 .. C23); the full blend with p33 (+0.2) already captures part of this, so realistic lift on top of v12x_blend is probably +0.0003..+0.0008. Caveat: harness control is z-only ctx (as in dw_v12sig).
Eval-time check: on c27, within-slate residual (p33 rank - v12 rank) correlates with pp12_cos (Spearman 0.07-0.08) and pp_prod11 (.07); linear R2 of the residual on cos+prod = 3.3% holdout. So the p33 model's added value over v12 is partly this PPMI centroid-product signal, but most of p33's difference is unexplained by it (probably its MLP/LightGBM interactions).

## Reusable code
`p33feat.py`: `pp_feats(D, X, rows, groups=('cos','rec','prod','user','multi'))` -> dict of float32 arrays (same contract as dw_v12sig `taste_metric`). Cuts: `python p33feat.py {23,25,27}` -> `cache/pp_c{cut}.parquet` (c27: train events ts<Jd(27) only, 686,824 train events, eval events only appended for own-history lookups; 529,390 rows built). Minimal winner set: groups=('cos','prod') -> `pp12_cos*`, `pp_prod0..11` (+ ppb12_cos). `test_pp.py`: reproduces c25 cache and shows features are unchanged when all post-cutoff labels are flipped (leak-safe).
Other files: `p33lib.py`, `a0.py`, `t2_probe.py`, `t3_spec.py`, `run_groups.py`, `results.jsonl`, `build.log`.
