# blending_v1 — blending / ensembling research for global AUC (written 2026-11-02, Dublin time)
Researcher: AUC Researcher subagent for Yeepee. No eval CSV used anywhere. Nothing uploaded. Scripts: research/blending_v1_scripts/ (screen.py, evnorm.py, logs).

## 0. Read first (state of the box differs from the older brief)
- Box shows public scores far above 0.818: v21fit/BLEND_RECIPE.md lists S1 = 0.829101, R1 = 0.829058, Q2 = 0.829016; base "EN" (ErBrown latest rank) = 0.828856. Our own models alone are ~0.819 public.
- Those 0.829 files are rank blends of EN (erbrown/merged.parquet, column `eb`) minus c*E0 plus our A/D/v8regen/v5. Weights were fit on PUBLIC scores (binormal model, v21fit/opt.py), not on local validation, and EN/E0 have no val25/val26 predictions, so they cannot be validated locally. I did not verify where `eb` comes from or whether it respects the "no evaluation csv" rule. Please have Eric/Grok Bot confirm provenance before relying on it. Public-score-fitted weights go against the "choose by local validation" rule.
- Grok Bot's own v21fit diagnostics (STATUS.md) already conclude reweighting existing signals gives only about +0.00003..+0.0001 on EN. My screen below agrees for our own members.

## 1. Inventory (aligned val preds, row order = v12/feat_v3/val{cut}.parquet; cached in cb_blend/oof_c{23,25,26}.npz by Code Buddy's blend_oof.py; alignment checks vs reported AUCs pass at 1e-5)
| member | what | C23 | C25 | C26 |
|---|---|---|---|---|
| z | v19 stack (b+d+2*mlp)/4 | .79679 | .81829 | none |
| M | ctl6 8 seeds (v21seq) | .79862 | .81904 | .82797 |
| big | v21seq_big 3/3/2 seeds | .79851 | .81916 | .82791 |
| list1 | v21list single seed | .79799 | .81857 | none |
| D | DCN-v2 bag (C23 is a proxy from v21dcn r1 configs) | .79793 | .81890 | none |
| ref | LightGBM ref484, 3 seeds | .79707 | .81778 | none |
| A | ctl6 + big pooled | .79864 | .81910 | .82798 |
All numbers measured (recomputed by me from the npz files). Only M/big/A exist at C26, so a true 3-cutoff stability test is only possible for M/big/A; 2-cutoff (C23<->C25) for the rest. Other npy member files exist (v9/ens, v11/ens, v19/ens...) but these are older/weaker.
No val preds exist for ErBrown EN/E0 or p33 (only eval-row preds), so those cannot be screened locally.

## 2. Measured screen (C23 / C25, global AUC; script screen.py, 2 threads, nice 19)
Best single: C23 M .79862, C25 big .81916.
Equal-weight blends (rows = member sets, logit mean / rank mean / prob mean):
| set | C23 logit / rank / prob | C25 logit / rank / prob |
|---|---|---|
| z+M | .79890 / .79889 / .79894 | .81916 / .81916 / .81907 |
| z+M+big | .79908 / .79908 / .79910 | .81929 / .81929 / .81926 |
| z+M+big+D | .79911 / .79911 / .79913 | .81952 / .81952 / .81949 |
| all 6 | .79908 / .79907 / .79910 | .81952 / .81951 / .81948 |
| z+M+big+D+list1 | .79904 / .79904 / .79905 | .81946 / .81945 / .81944 |
| M+big+D+list1 (no z) | .79873 / .79874 / .79874 | .81940 / .81939 / .81938 |
- Delta of best equal blend (z+M+big+D) vs best single: +0.00049 (C23), +0.00036 (C25). Everything beyond is noise-level.
- logit vs rank vs prob averaging: differences <= 0.00005 (a wash). Pick logit or rank; rank is safest if calibration offsets differ (z at C25 has mean logit -2.92 vs -3.45 for others; constant shifts are harmless in a linear logit blend).
- z (the diverse older stack) and D (DCN) are what add value; dropping z costs ~ -0.0004, adding list1 or ref adds nothing.
- Caruana forward selection (30 iterations, with replacement, AUC objective on one cutoff, tested on the other): fit C23->test C25 .81954 (rank .81952); fit C25->test C23 .79906 (rank .79905). That is equal to equal-weights (.81952/.79911) within ±0.00005. Selected weights are unstable across cutoffs (e.g. M .33 vs 0), which says only weight stability, not gain, is at stake.
- Nelder-Mead weight fit: INVALID, it did not move from the equal start (output weights = 1/6 exactly), so I report no result for it. Grok Bot's cb_blend/search.log has working NM fits: z,M fit C23 -> C25 .81923 vs fixed .4z+.6M .81921; i.e. the same flat result.
- Stacking meta-models fit on one cutoff, tested on the other, vs equal logit z+M+big+D:
  - LR on member logits (C=0.01): +0.00002 (C23->C25), -0.00005 (C25->C23). Null.
  - LR + event context (event max/mean/std/dev of blend): -0.00002 / -0.00003.
  - Shallow GBDT (7 leaves, 150 rounds, min_data 2000, l2 50) on member logits: -0.00014 / -0.00031. On members + event features: -0.00016 / -0.00030. On members + user activity (us_nev, tm_stale, hist_n_all, us_ctr): -0.00011 / -0.00022. All worse and the sign is consistent. A GBDT meta-model trained on one cutoff does not transfer to the other.
- Per-event-normalised components as blend inputs (evnorm.py; B = equal logit z+M+big+D; B + lam*(B - event mean / event logsumexp / event max)): every lam in {-0.2..0.4} is <= base on both cutoffs (best: lam=-0.1 on logsumexp, C23 .79910 vs .79911; C25 .81945 vs .81952). Null, consistent with the project convention not to use per-event normalisation.
Conclusion (measured): among our own members, blending is saturated at about +0.0004..+0.0005 over the best single model; no tested weighting, stacker or event-context scheme beats an equal logit/rank average of z+M+big+D on the held-out cutoff.

## 3. Ranked ideas
Gain column: "measured" = from section 2; everything else is an expectation, not a result.
1. Keep a simple equal-weight logit (or rank) average of DIVERSE members; add members only if they bring a new view (z, D worked; list1/ref did not). Effort low. Leak risk none. Measured +0.0004..0.0005 over best single. Sources: Caruana 2004 (forward selection ~ averaging), KDD Cup 2012 winners uniform-averaged top models, Kaggle Ensembling Guide. This is what Code Buddy's blend_oof.py already does; the recipe is z+M+big+D at equal weights.
2. New-VIEW members beat more seeds/same-family members. Every Criteo/Avazu/Outbrain/KDD12 winner blended structurally different models: FFM vs GBDT vs FTRL vs MF vs NN; Avazu 1st = 20 models incl. feature-subset models; KDD12 = classification, regression, ranking and MF models. Untried here: GBDT-leaf-embedding + FM/FFM-style model, an explicit FFM/FM on ID fields, an LDA/NMF k~12 view. Effort medium-high (a new model), leak risk none if train-only. Expected gain: this is the only lever that moved things (z and D each added ~+0.0002); gain not measured.
3. Cross-cutoff weight protocol: fit weights on C23, check on C25 and C26, then take the average of weights, shrunk 50% toward equal. Needed because Caruana/NM weights flip between cutoffs. Effort low. Gain: protects against loss, not a measured gain. Source: Caruana 2004 (overfitting of selection; bagged selection).
4. Second-level learner, only with true OOF members and rank:pairwise or LR (Outbrain 2nd/3rd/13th, Otto 1st, KDD12 non-linear blending). Measured here: LR null, GBDT negative when transferred across cutoffs. A stacker needs many cutoffs of OOF (we have 2-3, each a single day-ish snapshot). Do not spend time unless 5+ cutoffs of OOF exist. Leak risk none if OOF is train-only.
5. Post-blend calibration / mean shift: 3 Idiots subtracted 0.003 to match the leaderboard base rate. Under global AUC a constant shift is a no-op, so this transfers only if combined with an event/segment-dependent calibration; per-event normalisation measured null (section 2). Skip.
6. (Not recommended) Public-LB-fitted blend weights (KDD12 team did a leaderboard re-blend stage; Outbrain 2nd chose weights partly on public LB). That is exactly how the 0.829 files were fitted; it has big overfit risk with only 10 uploads/day and contradicts "choose by local validation".

## 4. Anonymous / ID-only CTR competitions: ensembling and feature tricks (all links opened as search results this session)
| contest | what won | transferable for us | source |
|---|---|---|---|
| Criteo Display Ad (1st, 3 Idiots) | GBDT leaf indices (30 trees depth 7; v2 50 trees) as features into FFM; infrequent categories -> one special tag (>0.0005 logloss gain); numerics -> categorical via floor(log(v)^2); instance-wise L2 normalisation; hashing to 1M dims; FFM+GBDT ~0.001 better than FFM alone | leaf embeddings, rare-ID bucketing, binning of counts, per-row L2 norm of sparse feature vector | https://www.kaggle.com/competitions/criteo-display-ad-challenge/writeups/3-idiots-3-idiots-solution-libffm ; slides https://www.pdf-archive.com/2015/10/30/kaggle-2014-criteo/ ; code https://github.com/ycjuan/kaggle-2014-criteo |
| Avazu (1st, 4 Idiots incl. Michael Jahrer) | ensemble of 20 models: 2 base, 2 bag-feature, 16 small subset models; private logloss 0.3797 vs 0.3813 for base alone; feature selection by dropping features on a 10% subsample; subset models kept only if ensemble improved | build many small diverse feature-subset models, keep each only if blend improves on holdout | https://www.kaggle.com/competitions/avazu-ctr-prediction/writeups/4-idiots-4-idiots-solution-libffm ; https://github.com/ycjuan/kaggle-avazu |
| Avazu 5th (Efimov) | FTRL-proximal with 3 different data sort orders + FM; hand weights 0.6 FM / 0.1 / 0.2 / 0.1 | seed/ordering diversity with a few hand-set weights | http://www.efimov-ml.com/pdfs/ClickThroughRate2015.pdf |
| KDD Cup 2012 Track 2 (1st, NTU Catch Up, AUC 0.8069 public/0.8089 private) | many diverse models (classification, regression, ranking, MF); stage 2 non-linear blending on a validation set plus re-blending; final = uniform average of top five RANK predictions, which beat test-set linear blending; best single 0.7968 -> blend 0.8062 -> 0.8069 | closest analog (global AUC, ID-heavy); uniform rank average of top-k is a strong, robust final step | https://www.csie.ntu.edu.tw/~htlin/paper/doc/wskdd12cup.pdf ; contest page https://www.kdd.org/kdd-cup/view/kdd-cup-2012-track-2 ; runner-up note https://blog.operasolutions.com/bid/223144/Opera-Solutions-Uses-Advanced-Analytics-to-Better-Predict-Click-Through-Rates-for-Search-Advertising |
| Outbrain CTR (2nd, Brain AFK/Darragh) | ~50 diverse FFM base models, level-2 XGBoost and Keras on out-of-fold preds with normalised time as an extra feature, final weighted geometric mean XGB^0.7*NN^0.3 | stacking works only with many OOF rows plus a time feature; geometric mean of two stackers | https://medium.com/kaggle-blog/outbrain-click-prediction-competition-winners-interview-2nd-place-team-brain-afk-darragh-610d072fa3fc |
| Outbrain (3rd) | FFM (incl. pairwise-rank libffm with sample weights) + XGBoost first layer, XGBoost stacking | pairwise-rank/weighted FFM as a distinct view | https://www.kaggle.com/competitions/outbrain-click-prediction/writeups/three-data-points-3rd-place-solution |
| Outbrain 13th (ololo) | SVM/FTRL + XGB/ET on mean-target-value features + FFM using XGB leaves, combined by XGB rank:pairwise; MTV computed on separate folds | MTV (target encoding) with fold separation; leaves must not be reused across folds | https://github.com/alexeygrigorev/outbrain-click-prediction-kaggle |
| Outbrain 19th write-up | weighted ARITHMETIC mean of LOGITS of 3 FFM + 1 FTRL beat other mean types; GBDT rank blender trained on validation only (not train, to avoid favouring overfit models) | logit mean is the right default; train blenders on held-out only | https://www.kdnuggets.com/2017/07/feature-engineering-help-kaggle-competition-3.html |
| Tencent Social Ads 2018 | LightGBM + DeepFM/NFFM, weighted ensemble tuned on validation AUC; cross-window counts/conversion ratios; 5-fold CV statistics to control leakage | count/ratio features with fold-controlled stats (we already do these) | https://www.besthub.dev/articles/winning-approach-of-the-tencent-advertising-algorithm-competition-feature-engineering-model-selection-and-future-work-6190ae94c5e8 ; https://github.com/nzc/tencent-contest |
Not sourced this session: Alimama, TalkingData, iPinYou and RecSys-challenge winner write-ups (I got no usable search results; I did not cite them).
Also cited for method: Caruana, Niculescu-Mizil, Crew, Ksikes, ICML 2004 https://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf (forward selection can optimise ROC Area directly); Otto 1st (Titericz & Semenov, 3-level stack, geometric blend) https://www.kaggle.com/competitions/otto-group-product-classification-challenge/writeups/gilberto-titericz-stanislav-semenov-1st-place-winn ; H&M 1st (retrieval + LightGBM/CatBoost blend, LightGBM best CV-to-LB correlation) https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/writeups/senkin13-30crmnsia-1st-place-solution ; MLWave Kaggle Ensembling Guide (rank averaging for AUC) https://github.com/MLWave/Kaggle-Ensemble-Guide .

## 5. Most transferable (ID-only tricks) for our pipeline
1. GBDT-leaf indices as sparse features into an FM/FFM-style or embedding model (Criteo 1st; Outbrain 13th). It is a different view, so it helps blending. Fit leaves on train only; refit per cutoff.
2. Rare-ID bucketing (count < k -> shared tag) and log^2 binning of counts (Criteo 1st, >0.0005 logloss). Check whether our item/user ID embeddings already do this.
3. Many small feature-subset models kept only if the held-out blend improves (Avazu 1st).
4. Uniform average of rank predictions of the top-k diverse models as the final step (KDD12 1st).
5. Train any blender on held-out/OOF rows only, with a time feature (Outbrain).

## 6. Recommended recipe to try next (all local, train-only, no eval rows)
a. Final blend = equal logit (or rank) mean of z, M, big, D (+ any new-view member). Measured: C23 .79911 / C25 .81952. Report C26 once z and D exist at C26 (they do not; Grok Bot must generate z and D val26 preds to verify the third cutoff).
b. New-view member to add (priority): GBDT-leaf + FM/embedding model, or LDA/NMF k~12 view. Accept it only if the equal blend improves on BOTH C23 and C25 (and C26 if available) by >= +0.0001 and rho to the blend < 0.97.
c. Weights: equal; or Caruana weights averaged across cutoffs and shrunk 50% to equal. Do not use GBDT meta-models.
d. Do not blend public-score-fitted weights into the final choice unless Eric accepts that risk; log provenance of EN/E0.
Run next: Code Buddy: rerun cb_blend/blend_oof.py with D at C23 from a real bag plus val26 for z/D/list1. Grok Bot: build the new-view member val23/25/26 + eval preds under the same seed convention. Data Wrangler: confirm provenance of erbrown `eb` and E0.

## 7. New-view members (follow-up, measured 2 Nov 2026, Europe/Dublin)
Question: is there a cheap blend member with a different view that lifts the equal logit blend z+M+big+D (C23 .79911, C25 .81952; C26 M+big+A .82798)? Script `blending_v1_scripts/newview.py CUT` (train-only: `lib20.load(cut)` = f-tables before the cutoff, scored on val{cut}; eval never touched; 2 threads, nice 19, seeds fixed; ~2-4 min per cutoff) and `nvblend.py CUT`. Predictions in `blending_v1_scripts/out/*_c{cut}.npy`, row order = `cb_blend/oof_c*.npz` (rowid/y asserted equal). Blend delta = AUC(blend with member) − AUC(base blend). "eq" = member gets an equal 1/(n+1) share (member rescaled to the base mean's std); w0.1 = 10% share. **All numbers MEASURED.**

### 7.1 (a) GBDT leaf indices → LR / FM (Criteo-1st style) — MEASURED, REJECTED
- What: LightGBM (100 trees, 31 leaves, lr .1, ff .5, top-200 features by v20 ref484 gain) trained on a random 50% of the train rows; `pred_leaf=True` on the other 50% and on val; one-hot (100 trees × 31 leaves = 3100 binary cols). Heads: sklearn LogisticRegression (C=.03/.3, lbfgs) and a pure-torch FM (k=8, linear + 2nd-order over 100 leaf fields, Adam 3e-3, bs 4096). No FFM library is installed (no xlearn/libffm/gensim/implicit); torch FM is the substitute. FM overfits after epoch 1 (C25 val AUC .8096→.8012→.7968→.7899), epoch-0 is the only usable point.
- Leakage risk: none beyond the existing feature tables (leaf model and LR both see only train rows; the GBDT/LR split avoids in-sample leaves). Effort: ~60 lines, already written.

| member | single C23 / C25 / C26 | rho vs blend (C23/C25/C26) | eq delta C23 / C25 / C26 | w0.1 delta C23 / C25 / C26 |
|---|---|---|---|---|
| leaf GBDT alone (val) | – / .81446 / – | – | – | – |
| leaf→LR C=.03 | .78726 / .81091 / .82142 | .949/.966/.968 | −.00045 / −.00039 / −.00011 | −.00009 / −.00010 / +.00009 |
| leaf→LR C=.3 | .78503 / .80996 / .82074 | .940/.963/.965 | −.00052 / −.00044 / −.00017 | −.00011 / −.00012 / +.00008 |
| leaf→FM ep0 | .79122 / .80958 / .82035 | .965/.963/.967 | −.00036 / −.00050 / −.00035 | −.00009 / −.00015 / 0.0 |
Verdict: rank corr to the blend is .94-.97 (not a new view: the leaves come from the same 250 tabular features) and the member is ~.008 AUC weaker than the best single, so any share ≥ 20% hurts. Only C26 at 10% shows +.00009, which is inside seed/bootstrap noise (~.0001) and has the wrong sign on C23/C25. Fails the acceptance rule (≥ +.0001 on BOTH cutoffs, rho < .97).
Caveat: this is a light version (half the train rows, 100 trees, top-250 features). The Criteo recipe uses full data and a tuned FFM; I judge a better leaf model unlikely to flip the sign because rho stays ≈ .96, but that is untested.

### 7.2 (b) NMF / LDA k≈12 taste view — ALREADY IN THE PIPELINE, not a new view (prior measurement, not rerun)
`v19/README.md` Exp 1: nmfU (user×item clicks), nmfP (PPMI item×item) and ldaU, k=12, features cos/aff/lift/z/dm/entropy, pruned to 24 cols (feat_NAs/NBs) and included in the 484-col ref set. Measured as features (dwbig LightGBM, delta vs v17cooc): A C25 +.00032, A C23 +.00048; blend C25 +.00027 (A). The signal is therefore already inside M/big/A/D and cannot be added again as a blend member. A direct-score NMF member (not features) is untested; expected to be weaker and highly correlated with `tB_dot_z` / `coA_pp12_*`, which are already the top gain features.

### 7.3 (c) Other high-diversity options — what the repo shows is already tried (all NO-GO or null)
- Feature-bagged GBDT (drop top-30 gain features, extra_trees, ff .3, 400 rounds lr .05; Avazu-style subset view) — MEASURED: single .79005/.81173/.82067, rho .964-.965, eq delta −.00059/−.00029/−.00031; REJECTED.
- Sequence models SASRec/GRU4Rec (`v16seq/NOTES.md`): best residual +.00004 on C25 vs +.0005 gate → NO-GO. Same for CatBoost with categorical sets (`v15cat/NOTES.md`, ≤ +.00003 as member), transductive eval-row aggregates (`v14_transductive/README.md`: null), trend/cold-start/two-hop/cluster/decayed-CTR features (`v21trend/README.md`: all ≤ noise, mean3 −.0001..−.0003), item2vec-equivalent PPMI-SVD embeddings (in v19/v17cooc).
- Not found in the repo and not screened (UNTESTED, no evidence either way): (i) a direct linear/FM model on raw sparse IDs (user-history-item × candidate-item crosses, hashed, Criteo/Avazu style) trained from the click log, used only as a blend member; I expect the same weak-single problem (v15cat: raw IDs overfit under the time split, val peaks after ~35 iterations); (ii) `v21graph/gcn.py` is being run by another agent; check its output before duplicating.

### 7.4 Conclusion and recommendation
1. No new-view member from (a)-(c) clears the bar; measured deltas are −.0001..−.0006 at equal weight. The ranked list is (best first): leaf→LR C=.03 (−.00045/−.00039/−.00011), leaf→FM ep0, gdrop, with NMF/LDA already integrated and everything else previously NO-GO.
2. Evidence from this and the previous screen is consistent: our own members saturate at about +.0004-.0005 over the best single (equal logit mean z+M+big+D); the remaining headroom is not in blending tabular-feature models. Use the equal-mean z+M+big+D (or +A) as the submission blend, and treat the 0.829 public files as unverifiable-provenance (ErBrown `eb`) rather than something a local blend can reproduce.
3. If Eric still wants a new-view attempt, the only untested candidate with a different information source is (c)(i), sparse hashed ID-cross FM from the click log; budget ~0.5 day, gate = ≥ +.0001 on both C23 and C25 at 10-20% share.

### 7.5 Run instructions (Data Wrangler / Code Buddy)
```
cd /workspace/huawei/research/blending_v1_scripts
# one heavy job at a time on the shared box; ~3.5 GB RAM peak, 2 threads
setsid nohup nice -n 19 ../../v20/.venv/bin/python newview.py 25 > nv25.log 2>&1 < /dev/null &   # then 23, then 26
../../v20/.venv/bin/python nvblend.py 25   # prints single AUC, rho, eq/w0.1/w0.2/w0.3 deltas; writes out/nvblend_c25.json
```
To test a different member: save its val raw scores as `out/<name>_c{cut}.npy` (float32, row order of `cb_blend/oof_c{cut}.npz`) and rerun `nvblend.py`. Weights are never fitted on eval; the only fixed shares tried are equal, .1, .2, .3. Note: `lib20.load` for cut 26 trains on f2..f5, as in v20 C26.

## 8. Feature-set 484 vs 490 (measured 2 Nov 2026, Europe/Dublin)
The 490 set = ref484 (`v20/feats_ref484.json`) + 6 transition-matrix features from `v12x/feat_tmx` (built by `v12x/mk/mkfeat.py`): `tm_inh, tm_cos, tm_cos_in, tm_cos_out, tm_cos_dm, tm_inh_n`. Also screened: 488 = 484 + 4 previously dropped `sq_pdot, sq_pcos, sq_mdot, sq_mcos`.
Recipe: v20/s0_ref.py config (lib20 DWBIG, 1000 rounds, seed 42), 2 threads, train-only, no eval rows. Script `blending_v1_scripts/fs490.py`, eval `fs490_eval.py` -> `out/fs490_eval.json`.

| Set | C23 single | d vs 484 | C25 single | d vs 484 | Blend4 delta (C23 / C25) |
|---|---|---|---|---|---|
| 484 (ref s0) | .79665 | - | .81748 | - | - |
| 490 (+tmx) | .79644 | -.00021 | .81756 | +.00009 | -.00011 / +.00002 |
| 488 (+sq4) | .79628 | -.00038 | .81745 | -.00002 | -.00014 / .00000 |

Verdict: the 6 missing features do not improve; 490 is a wash (one cutoff down .0002, the other up .0001, within seed noise ~1e-4), and adds nothing to the blend. Keep 484. C26 not run (one seed per config; ~30 min per run at 2 threads).
