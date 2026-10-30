# v19 — NMF topic features (Exp 1) + cross-slate pairwise (Exp 2), base v17cooc
Metric: global ROC AUC. B = TRANSDUCTIVE (train + eval histories in co-occurrence/NMF graph, no labels) = main; A = train-only (comparison only).
NEVER uploaded. No final packaged (C25 < 0.83 gate).

## Exp 1: NMF(12)/LDA features
Spaces: nmfU (user x item clicks, 1/sqrt weighting), nmfP (PPMI item x item), ldaU; features cos/aff/lift/z/dm/entropy, several profile decays; wu_rec. Script mkmf.py (60 cols), mksel.py (pruned 24 cols: all 15 nmfP + 9 nmfU). Final set = pruned (feat_NAs/feat_NBs).
Screen (dwbig LightGBM, 1000 rounds), delta vs v17cooc (C25 A .81719, B .81704):
| set | A C25 | A C23 | B C25 | B C23 |
| full 60 | +.00026 | +.00042 | +.00006 | +.00007 |
| pruned 24 | +.00032 | +.00048 | +.00038 | +.00014 |
Full blend C25 ((dwbig+dw_s+2*mean MLPx3)/4):
| | dwbig | dw_s | mlp3 | blend | v17cooc blend | delta |
| B | .81742 | .81713 | .81738 | .81830 | .81793 | +.00037 |
| A | .81751 | .81694 | .81741 | .81829 | .81802 | +.00027 |
Gate (+.00015) passed for both. Full-60 B weaker likely due to train/eval graph mismatch (train folds +13% added history vs +0.3% eval).

## Exp 2: cross-slate pairwise (xpair.py, LightGBM custom RankNet over random groups of m slates; within-slate share ~1/m), B, C25
| m | standalone | 
| 2 | .81615 (500r) |
| 6 | .81659 (400r; .81652 at 600) |
| 20 | .81650 (500r) |
Blend with nm B blend (z-score weights 0/.1/.2/.3/.4/.5): m=6 .81830/.81830/.81826/.81819/.81808/.81793; m=2,20 never above .81830. => no gain, NOT used. (Not run for A; mlp_pair.py not run.)

## Candidates (out/), 529,390 rows, rowids == v5, header rowid,predicted_probability, in [0,1], no NaN
sha256 of zips:
be8305756628ab967d9241fe370fdb786ed122255188366e2e593a249e159db3  probe_v19nm_A_trainonly_p33_w80.csv.zip
4ade1a04defa398bfaacb09ff9090398eae4db12767160c8a4b2df8503d2be55  probe_v19nm_B_TRANSDUCTIVE_p33_w80.csv.zip
3c96c79a92e6acb208587341d01d17da59b8e7a3ec2cdfd5e7b06c6d41ed323c  submission_v19nm_A_trainonly.csv.zip
c2bfae3de671b3177a4b87085562f6b08e78cb184856b9e642c042fac923ce5b  submission_v19nm_B_TRANSDUCTIVE.csv.zip
Probes = 0.8*rank(ours)+0.2*rank(p33) (same as v17cooc). Spearman B vs v17cooc B 0.9984. Probe C25 cannot be measured locally (p33 is eval-only).
Runtime: feature build (mkmf) + dwbig ~7 min + dw_s ~1.3 min + MLPx3 ~3x2-6 min on 8 threads; well within 15 min budget on 4 cores for the final-fit path was not timed on 4 cores.
Risks: gains ~+.0003 are ~5x the ±.00006 noise at C25 but small; B/train-fold graph mismatch; public v17cooc B probe only +.000032 over A, so expected public gain small.
Recommendation: upload probe_v19nm_B_TRANSDUCTIVE_p33_w80 first (public best family), then probe_v19nm_A_trainonly_p33_w80 if slots allow.
bundle v19_candidates.zip (4 zips + sha256 files): 2a3b300db9a0e6a4b21b6ffd28f084cb3ee7233329f4ab38d0a5098ad274af90
