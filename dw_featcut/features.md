# dw_featcut: feature inventory and cut list (Eric, Huawei TechArena Phase 1 CTR)

Written Fri 2026-10-30 (Europe/Dublin, UTC+0). Read-only analysis; nothing retrained at scale, nothing uploaded. All scripts/data in `/workspace/huawei/dw_featcut/`.

## 1. Which pipeline
The best public score (0.8182) is the v12x blend plus p33. It uses the **v12 435-column matrix** (dwbig LightGBM 1000r + dw_s 500r + 2×mean MLP, (dwbig+dw_s+2·mlp)/4 on logits). The same recipe has since been extended in two nested steps:

| set | columns | = | where |
|---|---|---|---|
| v12 | 435 | base v8-lite cache 377 + feat_sqC 16 + feat_dw 42 | v12/, v12x/ (the 0.8182 submission) |
| v17cooc | 460 | v12 + feat_Ap (co-occurrence cos, 25) | v17cooc/ |
| v19 nmAs | **484** | v17cooc + feat_NAs (NMF, 24) | v19/, current best local (C25 .81751 / C23 .79679) |

All numbers below are for the 484 set (it contains the other two). `--drop '^sq_(pdot|pcos|mdot|mcos)$'` is applied in every pipeline (the 4 raw columns are NOT in the 484: drift KS .09-.16; v10B dropping them gave +.0007/+.0005/+.0012 on C25/C23/C26). They are not counted below. The B variants (transductive) use the same columns with coB_/nmB_ names.

## 2. Inventory by family (484 columns)
Gain share = mean normalised LightGBM gain over 5 saved dwbig-recipe models (nmAs C23, C25 and nmBs C23, C25, C27; B columns mapped to A names). Only gain was saved (no split counts, no current model files).

| family | cols | gain share | safe cut | probably cut | keep | source dir | builder | what it is |
|---|---|---|---|---|---|---|---|---|
| feat_tt | 8 | 38.4% | 0 | 0 | 8 | `v11/feat_tt` | `v17run/src/ttfeat.py` | two-tower/learned topic-embedding scores of user history vs candidate (tA/tB dot, cos, lse, mean, within-event dmean/dmax/z) |
| feat_Ap | 25 | 17.2% | 0 | 0 | 25 | `v17cooc/feat_Ap (v17cooc/mkcooc.py)` | `v17run/src/cooc/mkcooc.py` | co-occurrence PPMI-embedding cos of history vs candidate (pp8/12/16 variants, ppb, x-variant), unit-norm, ucos_dot, 12 cos-products |
| feat_sqC | 16 | 9.0% | 0 | 0 | 16 | `v11/feat_sqC` | `v17run/src/seq11.py, avg_sq.py` | sequence-model (transformer) averaged logit/gap/attention stats; raw pdot/pcos/mdot/mcos pre-dropped by --drop |
| feat_dw | 42 | 6.7% | 1 | 0 | 41 | `v12/feat_dw` | `v17run/src/dw/mkdw.py` | DW features: impression/skip SVD cosines, joint cos, history-cos in impression space, click/impression centroids + slate coherence, c_* contrast features |
| feat_lr2 | 54 | 6.0% | 0 | 0 | 54 | `v11/feat_lr2` | `v17run/src/lrfeat2.py` | low-rank (SVD k=4..16) user-history vs item cosines, contrast variants (con05/con1), within-event dmax |
| feat_NAs | 24 | 3.3% | 5 | 0 | 19 | `v19/feat_NAs` | `v19/mkmf.py` | NMF(12) user-item and item-item-PPMI affinities/lifts/entropy/cos per-dim (variant A, train-only graph) |
| feat_x | 38 | 2.5% | 3 | 0 | 35 | `v11/feat_x` | `v17run/src/extra.py` | extra click-graph features: con/conr/crcos/icon/icos at k=32/64/256, cluster CTR + lift + imp counts |
| feat_pp | 9 | 0.9% | 0 | 0 | 9 | `v11/feat_pp` | `v17run/src/ppmi.py` | PPMI-embedding history cos / last3 / con05 features (+z, +rk) |
| v2_ | 108 | 6.7% | 32 | 5 | 71 | `base cache v8-lite (v9d/val/cache_mx)` | `v17run/src/build_folds.py (v2 dir)` | v2 behavioural stats: item imp/clk/ctr windows, ip (item-pair) counts+decays, user stats, user-item since/noclk, history len/gaps, co_* sums |
| ev_* | 88 | 5.3% | 16 | 4 | 68 | `feat_v3 / base cache` | `v17run/src/feats.py, v3feat_ne.py` | within-event transforms (rk, z, max, mean) of hist/co/sv/it/mf groups |
| sv_* | 17 | 1.6% | 0 | 0 | 17 | `feat_v3 / base cache` | `v17run/src/feats.py` | SVD (k=#) user-history / item cos, dot, norm, max, con |
| it_* | 17 | 1.2% | 5 | 2 | 10 | `feat_v3 / base cache` | `v17run/src/feats.py` | item impression/click counts, CTR, trends, decay |
| us_* | 13 | 0.8% | 7 | 2 | 4 | `feat_v3 / base cache` | `v17run/src/feats.py` | user event counts, CTR, history length, cold flags, recency |
| hist_* | 10 | 0.0% | 8 | 2 | 0 | `feat_v3 / base cache` | `v17run/src/feats.py` | user-item history counts/positions (n_all, noclk, since_*, in) |
| co_* | 12 | 0.1% | 6 | 4 | 2 | `feat_v3 / base cache` | `v17run/src/v3feat_ne.py` | user-history co-click cos / transition stats (max5, mean5, last, rw, sum5) |
| mf_* | 3 | 0.3% | 0 | 0 | 3 | `feat_v3 / base cache` | `v17run/src/feats.py` | matrix-factorisation score/dot/bias |
| **total** | 484 | 100% | 83 | 19 | 382 | | | |

(v3 families ev/sv/it/us/hist/co/mf = 160 columns, v2_ = 108, lr2+x+pp+tt = 109, all inside the 377-column base cache `v9d/val/cache_mx` joined by `cl.matrix()`.)

Top features by gain: 
- `tB_dot_z` (feat_tt) 27.1%
- `tB_dmean` (feat_tt) 7.5%
- `coA_pp12_cos_z` (feat_Ap) 4.0%
- `sq_logit` (feat_sqC) 3.2%
- `tB_dot` (feat_tt) 2.8%
- `coA_pp12_cos_dm` (feat_Ap) 2.6%
- `coA_pp12_cos` (feat_Ap) 1.8%
- `sq_z` (feat_sqC) 1.7%
- `coA_pp12x_cos_z` (feat_Ap) 1.3%
- `sq_gap` (feat_sqC) 1.1%
- `sq_gap2` (feat_sqC) 0.9%
- `coA_ppb12_cos` (feat_Ap) 0.9%
- `coA_pp_prod2` (feat_Ap) 0.7%
- `coA_ppb12_cos_z` (feat_Ap) 0.7%
- `v2_u_ctr` (v2_) 0.6%

## 3. Cut list
Tiers (a feature in the top-60 by gain is never cut):
- **safe (83)**: max gain share <2e-5 in all 5 runs (about zero), OR a |Spearman|>0.98 duplicate of a higher-gain column found by BOTH my analysis (300k rows, f2+f4 training tables, no eval rows) and v20/out/redundancy.json (agree on 76 of the 84/83 each).
- **probably safe (19)**: gain <1e-4 share in all 5 runs, or a duplicate found by only one of the two analyses.
- **keep (382)**: everything else, incl. all of feat_tt, feat_Ap, feat_sqC, feat_lr2, feat_pp.

Total: 102 columns (21%) flagged, carrying 2.3% (safe) + 0.3% (probably) of total gain. Note gain share is not harm: a duplicate's gain simply moves to its twin.

Sanity check, LightGBM 2 threads, C25 (train: 100k rows each from f3 and f4, = 200k; score 400k random val25 rows; lr .05, 127 leaves, ff .5, l2 30, 300 rounds, 2 seeds; niced under load ~9, so fit times are not reliable). Mean AUC over seeds:

| set | cols | seed | AUC@150 | AUC@300 | fit s |
|---|---|---|---|---|---|
| full | 484 | 0 | 0.81242 | 0.8096 | 125 |
| full | 484 | 1 | 0.81347 | 0.8105 | 137 |
| cut_safe | 401 | 0 | 0.81283 | 0.80961 | 126 |
| cut_safe | 401 | 1 | 0.81286 | 0.80982 | 122 |
| cut_safe+probably | 382 | 0 | 0.81266 | 0.80934 | 120 |
| cut_safe+probably | 382 | 1 | 0.81259 | 0.80966 | 63 |

Mean AUC@150: full .81295, safe cut .81285 (-.0001), safe+probably .81263 (-.0003). At 300 rounds (overfit regime on a 200k sample): .81005 / .80972 / .80950. Seed-to-seed noise in the full model alone is ~.001, so the deltas are inside the noise but the safe cut is consistently ~-.0001..-.0003, not positive. Conclusion: cutting does not help AUC either, it only saves columns; confirm on full folds before shipping (v20 chain is doing exactly this: s4_curve/s5_grid/s6_tiers with the same 484 ref).

### Safe to cut (83)
| feature | family | max gain share (5 runs) | rank | evidence |
|---|---|---|---|---|
| `co_tr_last` | co_* | 1.1e-05 | 470 | ~zero gain in all 5 runs |
| `co_tr_mean5` | co_* | 3.2e-05 | 452 | gain <1e-4 in all 5 runs; |Spearman|=1.000 with `co_tr_max5` (kept: higher gain); also dropped by v20 redundancy.json |
| `co_cos_sum20` | co_* | 9.8e-05 | 434 | gain <1e-4 in all 5 runs; |Spearman|=1.000 with `co_cos_max20` (kept: higher gain); also dropped by v20 redundancy.json |
| `co_cos_rw` | co_* | 1.1e-04 | 430 | |Spearman|=0.999 with `co_cos_max20` (kept: higher gain); also dropped by v20 redundancy.json |
| `co_tr_max20` | co_* | 9.9e-05 | 426 | gain <1e-4 in all 5 runs; |Spearman|=1.000 with `co_tr_mean20` (kept: higher gain); also dropped by v20 redundancy.json |
| `co_cos_mean20` | co_* | 1.3e-04 | 424 | |Spearman|=1.000 with `co_cos_max20` (kept: higher gain); also dropped by v20 redundancy.json |
| `ev_co_tr_last_rk` | ev_* | 9.2e-06 | 475 | ~zero gain in all 5 runs; |Spearman|=1.000 with `ev_co_tr_last_z` (kept: higher gain); also dropped by v20 redundancy.json |
| `ev_hist_noclk_mean` | ev_* | 1.3e-05 | 477 | ~zero gain in all 5 runs; |Spearman|=1.000 with `ev_hist_noclk_max` (kept: higher gain); also dropped by v20 redundancy.json |
| `ev_hist_n_all_rk` | ev_* | 1.5e-05 | 476 | ~zero gain in all 5 runs; |Spearman|=0.997 with `v2_uip_n_dm` (kept: higher gain); also dropped by v20 redundancy.json |
| `ev_co_tr_last_z` | ev_* | 1.8e-05 | 464 | ~zero gain in all 5 runs |
| `ev_hist_noclk_z` | ev_* | 4.3e-05 | 467 | gain <1e-4 in all 5 runs; |Spearman|=1.000 with `ev_hist_noclk_rk` (kept: higher gain); also dropped by v20 redundancy.json |
| `ev_co_tr_last_mean` | ev_* | 2.0e-05 | 456 | ~zero gain in all 5 runs; |Spearman|=1.000 with `ev_co_tr_last_max` (kept: higher gain) |
| `ev_co_cos_rw_rk` | ev_* | 2.3e-05 | 454 | gain <1e-4 in all 5 runs; |Spearman|=0.989 with `ev_co_cos_max20_z` (kept: higher gain); also dropped by v20 redundancy.json |
| `ev_co_tr_max5_rk` | ev_* | 2.5e-05 | 463 | gain <1e-4 in all 5 runs; |Spearman|=0.993 with `ev_co_tr_max5_z` (kept: higher gain); also dropped by v20 redundancy.json |
| `ev_hist_n_all_z` | ev_* | 4.8e-05 | 446 | gain <1e-4 in all 5 runs; |Spearman|=0.997 with `v2_uip_n_dm` (kept: higher gain); also dropped by v20 redundancy.json |
| `ev_hist_n_all_max` | ev_* | 5.9e-05 | 453 | gain <1e-4 in all 5 runs; |Spearman|=0.998 with `ev_hist_n_all_mean` (kept: higher gain); also dropped by v20 redundancy.json |
| `ev_co_cos_max20_rk` | ev_* | 5.4e-05 | 443 | gain <1e-4 in all 5 runs; |Spearman|=0.989 with `ev_co_cos_max20_z` (kept: higher gain); also dropped by v20 redundancy.json |
| `ev_co_tr_max5_max` | ev_* | 2.4e-04 | 391 | |Spearman|=1.000 with `ev_co_tr_max5_mean` (kept: higher gain); also dropped by v20 redundancy.json |
| `ev_co_cos_rw_max` | ev_* | 4.9e-04 | 353 | |Spearman|=0.996 with `ev_co_cos_rw_mean` (kept: higher gain); also dropped by v20 redundancy.json |
| `ev_it_ctr_z` | ev_* | 9.5e-04 | 298 | |Spearman|=0.986 with `v2_i_ctr_dm` (kept: higher gain); also dropped by v20 redundancy.json |
| `ev_it_ctr_mean` | ev_* | 1.1e-03 | 273 | |Spearman|=1.000 with `v2_ev_ictr_mean` (kept: higher gain); also dropped by v20 redundancy.json |
| `ev_it_clk_dec2d_mean` | ev_* | 1.2e-03 | 229 | |Spearman|=0.983 with `v2_ev_ipdec_mean` (kept: higher gain); also dropped by v20 redundancy.json |
| `nmA_nmfP_cos_e3` | feat_NAs | 9.4e-04 | 308 | |Spearman|=0.990 with `nmA_nmfP_cos_p1` (kept: higher gain); also dropped by v20 redundancy.json |
| `nmA_nmfP_cos_e10` | feat_NAs | 1.1e-03 | 309 | |Spearman|=0.985 with `nmA_nmfP_cos` (kept: higher gain); also dropped by v20 redundancy.json |
| `nmA_nmfU_cos_e10` | feat_NAs | 1.4e-03 | 163 | |Spearman|=0.986 with `nmA_nmfU_cos` (kept: higher gain); also dropped by v20 redundancy.json |
| `nmA_nmfP_cos_z` | feat_NAs | 1.5e-03 | 143 | |Spearman|=0.981 with `nmA_nmfP_cos_dm` (kept: higher gain); also dropped by v20 redundancy.json |
| `nmA_nmfP_aff` | feat_NAs | 1.8e-03 | 103 | |Spearman|=0.981 with `nmA_nmfP_lift` (kept: higher gain); also dropped by v20 redundancy.json |
| `c_skall` | feat_dw | 1.6e-03 | 159 | |Spearman|=0.982 with `c_impall` (kept: higher gain); also dropped by v20 redundancy.json |
| `x_con05_k64_rk` | feat_x | 1.2e-04 | 431 | |Spearman|=1.000 with `ev_sv_k64_con05_rk` (kept: higher gain); also dropped by v20 redundancy.json |
| `x_con05_k64_z` | feat_x | 5.6e-04 | 344 | |Spearman|=1.000 with `ev_sv_k64_con05_z` (kept: higher gain); also dropped by v20 redundancy.json |
| `x_con05_k64` | feat_x | 4.7e-04 | 349 | |Spearman|=1.000 with `sv_k64_con05` (kept: higher gain); also dropped by v20 redundancy.json |
| `hist_in` | hist_* | 0.0e+00 | 481 | ~zero gain in all 5 runs; |Spearman|=1.000 with `hist_n_all` (kept: higher gain); also dropped by v20 redundancy.json |
| `hist_n_win` | hist_* | 5.0e-06 | 481 | ~zero gain in all 5 runs |
| `hist_imp` | hist_* | 6.0e-06 | 481 | ~zero gain in all 5 runs; |Spearman|=1.000 with `v2_ui_imp` (kept: higher gain); also dropped by v20 redundancy.json |
| `hist_noclk` | hist_* | 8.6e-06 | 474 | ~zero gain in all 5 runs; |Spearman|=1.000 with `v2_ui_noclk` (kept: higher gain); also dropped by v20 redundancy.json |
| `hist_n_warm` | hist_* | 1.6e-05 | 469 | ~zero gain in all 5 runs |
| `hist_pos` | hist_* | 2.5e-05 | 462 | gain <1e-4 in all 5 runs; |Spearman|=1.000 with `v2_inh_pos` (kept: higher gain); also dropped by v20 redundancy.json |
| `hist_n_own` | hist_* | 2.3e-05 | 460 | gain <1e-4 in all 5 runs; |Spearman|=0.995 with `hist_n_all` (kept: higher gain); also dropped by v20 redundancy.json |
| `hist_since_clk` | hist_* | 4.6e-05 | 445 | gain <1e-4 in all 5 runs; |Spearman|=1.000 with `v2_inh_since` (kept: higher gain); also dropped by v20 redundancy.json |
| `it_nclk_all` | it_* | 6.5e-04 | 331 | |Spearman|=0.992 with `v2_ip_users` (kept: higher gain); also dropped by v20 redundancy.json |
| `it_nusers` | it_* | 8.1e-04 | 289 | |Spearman|=1.000 with `v2_ip_users` (kept: higher gain); also dropped by v20 redundancy.json |
| `it_imp` | it_* | 1.0e-03 | 277 | |Spearman|=1.000 with `v2_i_imp` (kept: higher gain); also dropped by v20 redundancy.json |
| `it_clk_dec2d` | it_* | 1.2e-03 | 227 | |Spearman|=0.993 with `v2_ip_dec3` (kept: higher gain); also dropped by v20 redundancy.json |
| `it_imp_rate` | it_* | 1.4e-03 | 173 | |Spearman|=1.000 with `v2_i_imp_rate` (kept: higher gain); also dropped by v20 redundancy.json |
| `us_nohist` | us_* | 0.0e+00 | 481 | ~zero gain in all 5 runs |
| `us_cold` | us_* | 0.0e+00 | 481 | ~zero gain in all 5 runs |
| `us_hlen` | us_* | 1.2e-04 | 422 | |Spearman|=1.000 with `v2_h_len` (kept: higher gain); also dropped by v20 redundancy.json |
| `us_nclk_all` | us_* | 3.5e-04 | 355 | |Spearman|=0.988 with `v2_up_uniq` (kept: higher gain); also dropped by v20 redundancy.json |
| `us_nuniq` | us_* | 6.2e-04 | 335 | |Spearman|=1.000 with `v2_up_uniq` (kept: higher gain); also dropped by v20 redundancy.json |
| `us_nev` | us_* | 1.1e-03 | 329 | |Spearman|=1.000 with `v2_u_ev` (kept: higher gain); also dropped by v20 redundancy.json |
| `us_since_clk` | us_* | 8.6e-04 | 310 | |Spearman|=1.000 with `v2_h_since_last` (kept: higher gain); also dropped by v20 redundancy.json |
| `v2_ui_clk` | v2_ | 0.0e+00 | 481 | ~zero gain in all 5 runs; |Spearman|=1.000 with `hist_n_win` (kept: higher gain); also dropped by v20 redundancy.json |
| `v2_tr_last` | v2_ | 7.6e-06 | 478 | ~zero gain in all 5 runs; |Spearman|=1.000 with `v2_trp_last` (kept: higher gain); also dropped by v20 redundancy.json |
| `v2_ui_imp` | v2_ | 5.3e-06 | 473 | ~zero gain in all 5 runs |
| `v2_trp_last` | v2_ | 1.8e-05 | 472 | ~zero gain in all 5 runs |
| `v2_ui_noclk_dm` | v2_ | 4.2e-05 | 471 | gain <1e-4 in all 5 runs; |Spearman|=1.000 with `ev_hist_noclk_rk` (kept: higher gain); also dropped by v20 redundancy.json |
| `v2_tr_last3` | v2_ | 2.1e-05 | 459 | gain <1e-4 in all 5 runs; |Spearman|=1.000 with `v2_trp_last3` (kept: higher gain); also dropped by v20 redundancy.json |
| `v2_ev_n_inh` | v2_ | 4.2e-05 | 461 | gain <1e-4 in all 5 runs; |Spearman|=0.991 with `ev_hist_n_all_mean` (kept: higher gain); also dropped by v20 redundancy.json |
| `v2_tr_last3_rk` | v2_ | 3.9e-05 | 450 | gain <1e-4 in all 5 runs; |Spearman|=0.992 with `v2_tr_last3_dm` (kept: higher gain); also dropped by v20 redundancy.json |
| `v2_inh_cnt` | v2_ | 3.5e-05 | 451 | gain <1e-4 in all 5 runs; |Spearman|=0.995 with `hist_n_all` (kept: higher gain); also dropped by v20 redundancy.json |
| `v2_ui_noclk_rk` | v2_ | 5.9e-05 | 457 | gain <1e-4 in all 5 runs; |Spearman|=1.000 with `ev_hist_noclk_rk` (kept: higher gain); also dropped by v20 redundancy.json |
| `v2_uip_n_rk` | v2_ | 5.8e-05 | 444 | gain <1e-4 in all 5 runs; |Spearman|=0.997 with `v2_uip_n_dm` (kept: higher gain); also dropped by v20 redundancy.json |
| `v2_ip_1d` | v2_ | 4.1e-05 | 448 | gain <1e-4 in all 5 runs; |Spearman|=1.000 with `it_clk24` (kept: higher gain); also dropped by v20 redundancy.json |
| `v2_co_nz` | v2_ | 4.6e-05 | 442 | gain <1e-4 in all 5 runs; |Spearman|=0.994 with `v2_co_rw` (kept: higher gain); also dropped by v20 redundancy.json |
| `v2_co_cnt` | v2_ | 6.3e-05 | 438 | gain <1e-4 in all 5 runs; |Spearman|=0.994 with `v2_co_rw` (kept: higher gain); also dropped by v20 redundancy.json |
| `v2_i_clk1d` | v2_ | 8.4e-05 | 436 | gain <1e-4 in all 5 runs; |Spearman|=1.000 with `it_clk24` (kept: higher gain); also dropped by v20 redundancy.json |
| `v2_co_max_rk` | v2_ | 1.3e-04 | 435 | |Spearman|=0.991 with `v2_co_rw_rk` (kept: higher gain); also dropped by v20 redundancy.json |
| `v2_co_sum_rk` | v2_ | 1.4e-04 | 427 | |Spearman|=0.994 with `v2_co_rw_rk` (kept: higher gain); also dropped by v20 redundancy.json |
| `v2_uip_since` | v2_ | 1.3e-04 | 418 | |Spearman|=1.000 with `v2_inh_since` (kept: higher gain); also dropped by v20 redundancy.json |
| `v2_up_1d` | v2_ | 1.6e-04 | 413 | |Spearman|=1.000 with `us_nclk24` (kept: higher gain); also dropped by v20 redundancy.json |
| `v2_co_sum_rmax` | v2_ | 1.6e-04 | 419 | |Spearman|=0.991 with `v2_co_rw` (kept: higher gain); also dropped by v20 redundancy.json |
| `v2_co_rw_rmax` | v2_ | 1.6e-04 | 414 | |Spearman|=0.995 with `v2_co_rw` (kept: higher gain); also dropped by v20 redundancy.json |
| `v2_i_ctr_rk` | v2_ | 2.2e-04 | 396 | |Spearman|=1.000 with `ev_it_ctr_rk` (kept: higher gain); also dropped by v20 redundancy.json |
| `v2_co_sum` | v2_ | 1.9e-04 | 395 | |Spearman|=0.995 with `v2_co_rw` (kept: higher gain); also dropped by v20 redundancy.json |
| `v2_co_mean` | v2_ | 2.6e-04 | 374 | |Spearman|=0.996 with `v2_co_rw` (kept: higher gain); also dropped by v20 redundancy.json |
| `v2_co_max` | v2_ | 2.9e-04 | 368 | |Spearman|=0.993 with `v2_co_rw` (kept: higher gain); also dropped by v20 redundancy.json |
| `v2_ip_n_rk` | v2_ | 3.5e-04 | 359 | |Spearman|=1.000 with `ev_it_nclk_all_rk` (kept: higher gain); also dropped by v20 redundancy.json |
| `v2_up_since` | v2_ | 6.3e-04 | 347 | |Spearman|=1.000 with `v2_h_since_last` (kept: higher gain); also dropped by v20 redundancy.json |
| `v2_up_n` | v2_ | 5.6e-04 | 341 | |Spearman|=0.988 with `v2_up_uniq` (kept: higher gain); also dropped by v20 redundancy.json |
| `v2_i_ctr` | v2_ | 6.1e-04 | 339 | |Spearman|=1.000 with `it_ctr` (kept: higher gain); also dropped by v20 redundancy.json |
| `v2_i_ctr1d` | v2_ | 7.9e-04 | 332 | |Spearman|=1.000 with `it_ctr24` (kept: higher gain); also dropped by v20 redundancy.json |
| `v2_ip_warm` | v2_ | 6.9e-04 | 325 | |Spearman|=1.000 with `it_nclk_warm` (kept: higher gain); also dropped by v20 redundancy.json |
| `v2_ip_n` | v2_ | 7.9e-04 | 316 | |Spearman|=0.992 with `v2_ip_users` (kept: higher gain); also dropped by v20 redundancy.json |

### Probably safe (19)
| feature | family | max gain share (5 runs) | rank | evidence |
|---|---|---|---|---|
| `co_tr_max5` | co_* | 4.2e-05 | 449 | gain <1e-4 in all 5 runs |
| `co_cos_last` | co_* | 7.8e-05 | 437 | gain <1e-4 in all 5 runs |
| `co_cos_mean5` | co_* | 1.4e-04 | 412 | |Spearman|=1.000 with `co_cos_max5` (kept: higher gain) |
| `co_cos_max5` | co_* | 1.6e-04 | 417 |  |
| `ev_hist_noclk_max` | ev_* | 2.0e-05 | 466 | gain <1e-4 in all 5 runs |
| `ev_co_tr_last_max` | ev_* | 2.6e-05 | 465 | gain <1e-4 in all 5 runs |
| `ev_hist_noclk_rk` | ev_* | 7.9e-05 | 455 | gain <1e-4 in all 5 runs |
| `ev_hist_n_all_mean` | ev_* | 3.7e-05 | 447 | gain <1e-4 in all 5 runs |
| `hist_since_imp` | hist_* | 4.1e-05 | 458 | gain <1e-4 in all 5 runs; |Spearman|=1.000 with `v2_ui_since_imp` (kept: higher gain) |
| `hist_n_all` | hist_* | 9.8e-05 | 441 | gain <1e-4 in all 5 runs |
| `it_clk` | it_* | 2.3e-04 | 400 |  |
| `it_imp24` | it_* | 7.8e-04 | 313 | |Spearman|=1.000 with `v2_i_imp1d` (kept: higher gain) |
| `us_nclk_win` | us_* | 2.2e-04 | 408 | |Spearman|=1.000 with `v2_u_clk` (kept: higher gain) |
| `us_since_ev` | us_* | 9.6e-04 | 279 | |Spearman|=1.000 with `v2_u_since_ev` (kept: higher gain) |
| `v2_ui_noclk` | v2_ | 2.0e-05 | 468 | gain <1e-4 in all 5 runs |
| `v2_uip_n` | v2_ | 9.6e-05 | 439 | gain <1e-4 in all 5 runs; |Spearman|=1.000 with `hist_n_all` (kept: higher gain) |
| `v2_u_clk` | v2_ | 2.8e-04 | 415 |  |
| `v2_i_clk` | v2_ | 2.3e-04 | 399 | |Spearman|=1.000 with `it_clk` (kept: higher gain) |
| `v2_i_imp1d` | v2_ | 8.0e-04 | 317 |  |

### Keep (evidence)
- feat_tt (8 cols): 38% of gain; `tB_dot_z`, `tB_dmean` are top-2. Removing tt costs -.0057..-.0082 AUC (ablations).
- feat_Ap (coA, 25): 17% gain; cos/prod columns are strong, 0 cut.
- feat_sqC (16): 9% gain; sq_logit/z/gap/gap2 lead. The 4 raw dot/cos columns are already dropped.
- feat_dw (42, 1 cut), feat_lr2 (54, 0), feat_pp (9, 0): no cuttable columns (lr2: only one pair above 0.95, rho .954, so nothing is redundant at the 0.98 line; 122 pairs sit at .95-.98 across all families and were left).
- ev_* (88; 20 cut): the v3 ablation gave ev +.0065, so cut only the 20 flagged zero-gain ones, not the family.
- NMF per-dim and LDA columns were already pruned in v19 (cost nothing).

### Already excluded; do not re-add (earlier negative results)
prop/tm/tmx, fr/au/uc/ty, GRU/SAS sequence models (v16), XGB/lambdarank/CatBoost/max_bin variants (v12x, v15), sig features (dw_v12sig), lifecycle, repeat propensity, p33 as feature (dw_p33), classic seq (v18classic), low-rank extras in dw_lb. See v12x/NOTES.md, dw_v12sig/findings.md, dw_p33/findings.md, dw_lb/findings.md.

## 4. Expected saving
- Columns: 484 -> 401 (safe only, -17%) or 382 (safe + probably, -21%).
- LightGBM training/inference time scales roughly with column count (ff .5, col-wise histograms): dwbig 321 s (v12 note) -> ~265-255 s; dw_s 74 s -> ~60 s; MLP input smaller, ~5-10%. About 1-1.5 min per full v12-style run, ~15-20% of the model-fit stages. Feature build time is NOT saved by a regex drop (columns are built in the base cache and ev_*/v2_ stages); to save that the builders would have to be edited.
- Matrix memory 484x4 B x ~4.4M rows: -0.3 GB.
- Expected AUC change: about 0 to -.0003 on the small sample; not proven positive. Honest take: this is a speed/memory cleanup, not an AUC gain. The real accuracy lever remains tt/coA/sq.

## 5. How to use
- `drop_regex_safe.txt` or `drop_regex_safe_plus_probably.txt`: one line, `re.search` regex `^(a|b|...)$` over column names (verified: matches exactly 83 / 102 of the 484). ens.py takes a single --drop, so join with the existing one using `|`: `--drop '^sq_(pdot|pcos|mdot|mcos)$|<regex>'`.
- `drop_list.txt`: plain column list with # comments.
- For B-variant models use the same names but replace `coA_`->`coB_` and `nmA_`->`nmB_` (5 cuts are nmA_ columns, 0 are coA_; the B models' nmB_ names need the same cuts).
- Supporting: `gain_table.csv` (per-feature gain in 5 runs, rank, family, source, redundancy partner, tier), `redundant_dropped.json`, `corr_out.json`, `check_results.json`, `sample300k.parquet`.
