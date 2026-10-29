# Analog competitions for Huawei TechArena 2026 CTR (team Yeepee)
Written 2026-10-29 (Europe/Dublin). Web research only, no compute. Browsing went through the box network (Eric's computer was offline).

**Our setup:** slate of 10 candidates per event, at most one click (37% no click), ID-only, <=20-item click history with timestamps, eval = each user's first event after a cutoff (0-6 days stale), global row-level ROC AUC. Data dig: ~12-dim latent taste, conditional-logit-like with an outside option, weak popularity, random IDs.
**Already tried (not repeated here):** counts/CTR/recency/user x item crosses, PPMI-SVD k=12 + coordinate products, SVD-init two-tower softmax, DIN/SASRec/GRU, within-slate rank/z, LGBM+MLP blend, lambdarank (worse), event-level decomposition, co-visitation, count-drift normalisation. Also already in the notes: EASE (auc_ideas_v1), slot/position (null, data_dig_v1).
**Rule respected:** every idea below is train-only; none needs aggregates over eval rows.

**Honest framing:** no public competition is an exact analogue (all have metadata or real structure; ours is synthetic ID-only). The nearest *structure* is MIND / Yandex PWS / Outbrain / Trivago (impression lists + click history); the nearest *metric* is KDD Cup 2012 T2, TalkingData, DIGIX, MIND (AUC).

**URL status legend:** [200] fetched/curl-checked this session. [403*] blocked to scripts, existence confirmed by search. Dropped: layer6ai-labs/RecSys2019 repo (404), recsyschallenge.com/2015 (404), efimov PDF (500), researchr RomovS15 (no response), Kaggle blog Dataiku post (500), MS-Research Dataiku PDF (500).
"(read)" = I read the page/PDF text. "(snippet)" = only a search summary, treat as less certain.

---
## Ranking by closeness

| # | Competition | Slate? | ID-only? | History? | Metric | Closeness |
|---|---|---|---|---|---|---|
| 1 | Yandex Personalized Web Search (2013-14) | 10 results/query | mostly IDs (user, query, url, domain) | yes (user click logs) | NDCG@10 | very high (structure) |
| 2 | Outbrain Click Prediction (2016-17) | 2-12 ads/display, 1 clicked | IDs + weak meta | page views | MAP@12 | high |
| 3 | MIND news rec (MSNews, 2020) | impression list, clicked/non | IDs + text | click history | per-impression AUC / MRR / nDCG | high (but text) |
| 4 | RecSys Challenge 2019 (Trivago) | impression list (25), 1 clickout | metadata | session | MRR | high (structure) |
| 5 | KDD Cup 2012 Track 2 (Tencent Soso ads) | ads shown per query | mostly IDs + tokens | user click logs | **global AUC** | high (metric) |
| 6 | Avito Context Ad Clicks (2015) | ads per search | IDs + meta | user history | logloss | med |
| 7 | Huawei DIGIX CTR 2020 | single row | IDs + categorical | behavior logs | AUC | med (same organiser) |
| 8 | Expedia ICDM 2013 hotel ranking | search result list | metadata | weak | NDCG@38 | med |
| 9 | OTTO (Kaggle 2022-23) | none (session next-item) | **pure IDs** | yes | Recall@20 | med (ID-only) |
| 10 | H&M (Kaggle 2022) | none | metadata | yes | MAP@12 | low-med |
| 11 | TalkingData AdTracking (2018) | none | categorical IDs | click streams | AUC | low-med (AUC, IDs) |
| 12 | RecSys 2015 (YooChoose) | none | IDs | session | custom F1-ish | low-med |
| 13 | Avazu / Criteo | none | hashed cats | none | logloss | low |
| 14 | Coveo SIGIR eCom 2021, Dressipi RecSys 2022, ShareChat RecSys 2023, Tencent 2018 | none | mixed | session | various | low |

---
## TOP 5 IN DETAIL

### 1. Yandex Personalized Web Search Challenge (Kaggle, 2013-14)
- Comp: https://www.kaggle.com/c/yandex-personalized-web-search-challenge [200]
- Similarity: re-rank the **10 results** of a query using the user's own click history (IDs of query/url/domain); test = a last session per user after the training period, so the same "stale user, first event after cutoff" shape. Metric differs: NDCG@10 (within-list), signals are satisfaction levels. No outside option (user may click nothing), but "missed/skipped" labels model it.
- Top solution write-ups:
  - Dataiku slides (read): https://www.slideshare.net/slideshow/dataiku-pap-is-2014-winning-kaggle-yandex/42476320 [200]
  - Dataiku paper: https://www.academia.edu/9872579/Dataikus_Solution_to_Yandexs_Personalized_Web_Search_Challenge [403*]
  - What I read: random stop point in train to mimic the test cut, features from user history at that point, GBDT/pairwise LTR blends, tuned hyperparameters (large min-leaf) directly against the metric. (snippet) labels: satisfied 2/1/0, "missed" (below last click) vs "skipped" (above last click); pairwise LTR; NDCG@10 0.80647 vs baseline 0.79056.
- Untried ideas:
  1. **Mimic-test sampling of the training cut**: build train rows by choosing, per user, a random "stop point" and computing history only from before it, so train history staleness ~ eval's 0-6 d distribution. Relevance: **med** (we already did eval-like folds for validation; do check the *training* rows use the same stale-history distribution rather than only validation).
  2. **Skip/miss evidence**: treat past slates where the item was shown and not clicked as negative evidence for that user x item (and for neighbours in factor space). Relevance: **low-med** (likely covered by user x item impression crosses, per notes).

### 2. Outbrain Click Prediction (Kaggle, 2016-17)
- Comp: https://www.kaggle.com/competitions/outbrain-click-prediction/overview [200]
- Similarity: display_id = a slate of 2-12 candidate ads, exactly one clicked; user page-view history. Metric MAP@12 (within-slate), not global AUC. Real metadata (docs/topics) exists, ours doesn't.
- Write-ups:
  - 3rd place (Little Boat/jiweiliu/xiaozhouwang) (read, incl. comments): https://www.kaggle.com/competitions/outbrain-click-prediction/writeups/three-data-points-3rd-place-solution [200]
  - 2nd place interview (Brain-AFK/Darragh): https://medium.com/kaggle-blog/outbrain-click-prediction-competition-winners-interview-2nd-place-team-brain-afk-darragh-610d072fa3fc [403*]. (snippet) key trick: the other ads in the same display set as features.
  - Code: https://github.com/alexeygrigorev/outbrain-click-prediction-kaggle/ [200]
- Read facts (3rd place): FFM + XGBoost (rank:pairwise) first layer, XGB second layer. **"Present/future" split**: separate models for in-time vs future-time test rows, `sample weight = 1 + 2*0.5^(days_before_last_day)` for the "future" model (+0.001 MAP single, ~+0.002 with timestamp feature in the present model). **GBDT leaves as FFM features**: single model no gain (0.69586 -> 0.69588), but the weighted average of the two scored 0.6971 (+0.0012). CTR-with-threshold features overfit, so they dropped them. Pairwise FFM beat pointwise by ~0.0015 in a commenter's test.
- Untried ideas:
  1. **Recency-weighted training** for the stale-eval population (exp-decay sample weight on training cutoff age, or train a separate "future/stale" model weighted toward long-gap rows). Relevance: **med** (cheap; our drift tests were flat for counts but not for sample weights).
  2. **GBDT leaf indices -> second model** (leaf-one-hot -> embedding/SVD -> MLP or linear/FM member). Relevance: **med** (blend diversity; Layer 6 2019 got +0.3 MRR points from the same idea, see #4).
  3. **Competitor-composition features** (candidate x the other 9 candidates, e.g. FM/FFM with competitor IDs or "similarity of candidate to the slate's strongest other candidate"). Relevance: **low-med** (our softmax two-tower already conditions on the slate; check marginal value before spending time).

### 3. MIND (Microsoft News, 2020) - impression-based
- Comp: https://msnews.github.io/competition.html [200] ; CodaLab: https://codalab.lisn.upsaclay.fr/competitions/420 [200]
- Similarity: **impression = list of candidates marked clicked/not clicked**, with a user click history; metric = mean per-impression AUC (+MRR, nDCG). Test is a week after train, so there is a **temporal shift**. Differences: news text drives it; IDs alone are near chance in the official baselines.
- Write-ups (all read):
  - Winning-team "bag of tricks" report: https://msnews.github.io/assets/doc/2.pdf [200]
  - Other team reports: https://msnews.github.io/assets/doc/3.pdf [200] , https://msnews.github.io/assets/doc/4.pdf [200]
  - Chinese-language summary: https://mp.weixin.qq.com/s/7_lSvNrCc_hJPb6giN7OJQ [200]
- Read facts: engineered features and a two-stage pipeline that **raised dev AUC hurt the test set one week ahead**, notably popularity/recency features (shift!). Ensembling by averaging **scaled/unnormalised softmax probabilities** beat rank and logit averaging; bagging over differently built training sets; snapshot ensembling; fine-tuning on the validation days helped because they are closer to the test period.
- Untried ideas:
  1. **Blend in probability space and compare to logit/rank averaging** for NN bags (and weights by validation, not equal). Relevance: **med** (zero-cost test; global AUC may prefer one form).
  2. **Shift audit**: ablate every recency/popularity-derived feature on a stale-gap-matched fold; keep only features whose gain survives the horizon. Relevance: **high** as a safeguard (we already saw count-drift flat), low as new signal.
  3. **Final fit on the most recent window + snapshot/bagged NN seeds**. Relevance: **low-med**.

### 4. RecSys Challenge 2019 (Trivago): session hotel re-ranking
- Comp: https://www.recsyschallenge.com/2019/ [200] ; dataset paper https://dl.acm.org/doi/fullHtml/10.1145/3412379 [403*]
- Similarity: **impression list (up to 25) per clickout, one item clicked**, session history; MRR. Real hotel metadata (price/stars) dominates in the winners, which we lack.
- Write-ups:
  - Layer 6, 2nd place (full text read): https://www.cs.toronto.edu/~mvolkovs/recsys2019_challenge.pdf [fetched OK by WebFetch; curl timed out] ; DOI https://doi.org/10.1145/3359555.3359558 [403*] ; page https://layer6.ai/publications/acm-recsys-challenge-2019/ [200]
  - LogicAI, 1st place: code https://github.com/logicai-io/recsys2019 [200] ; paper https://dl.acm.org/doi/abs/10.1145/3359555.3359557 [403*]
- Read facts (Layer 6): 330 features per impression item; **independent binary cross-entropy beat ranking and softmax objectives** for both GBM and DL (agrees with our lambdarank result); **ordinal/relative-rank features beat absolute ones**; features that summarise all impression items and contrast the target against them; transformer over the impression list with position embeddings; **GBM leaf indices -> randomized SVD (128-d) -> fed to the NN gave > +3 MRR points vs raw features in their NN**; val MRR LSTM 0.6721, Transformer 0.6734, +SVD leaves 0.6750/0.6765, GBM 0.6771, **linear blend 0.6798**; out-of-time validation matched LB closely.
- Untried ideas:
  1. **Joint slate transformer** (self-attention across the 10 candidates + history tokens, per-item BCE, shared FFN head) as a blend member. Relevance: **med** (our two-tower is item-vs-user; attention lets candidates compete explicitly, though the true generator is a plain logit, so a gain is not assured).
  2. **Leaf-SVD input for the MLP** (same as Outbrain idea 2). Relevance: **med**.
  3. **Yeo-Johnson + sign(x)*embed(|x|) for signed rank-differences** for NN inputs. Relevance: **low**.

### 5. KDD Cup 2012 Track 2 (Tencent Soso, ad CTR; metric AUC)
- Comp: https://www.kaggle.com/c/kddcup2012-track2 [200]
- Similarity: **the only one scored with a global AUC**; user/ad/query IDs, temporal split, many cold-start users. Different: click prediction per ad impression, tokens exist, large click-rate counts.
- Write-ups (read the NTU one):
  - NTU team (winner): https://www.csie.ntu.edu.tw/~htlin/paper/doc/wskdd12cup.pdf [200]
  - Another top team (Jyun-Yu Jiang): https://jyunyu.csie.org/docs/pubs/kddcup2012paper.pdf [200]
- Read facts (NTU paper): the validation set was built to **mimic the test set's cold-start-user ratio / time split**; many diverse models (MF variants, RF, GBDT, FM) blended with a **RankNet / pairwise blender** after score transformation; some **MF models trained with pairwise ranking loss** to target AUC; Good-Turing-style back-off smoothing generalised better than additive smoothing for sparse counts; test-set linear blending using leaderboard AUC feedback (not recommended: uses LB probing).
- Untried ideas:
  1. **Global pairwise-AUC objective**: sampled positive-negative pairs across *different* events (not within-event lambdarank). Implementation options: NN/linear stage with RankNet loss on random cross-event pairs, or XGBoost `rank:pairwise` with random bucket groups mixing positives and negatives. Relevance: **med** (our within-event lambdarank was worse, but it optimises the wrong thing for a global AUC; BCE is only a proxy).
  2. **Back-off smoothing (Good-Turing / hierarchical shrinkage to a user-cluster or item-cluster prior) instead of additive/EB constants** for sparse user x item stats. Relevance: **low-med** (EB CTR is already tried).
  3. **Pairwise-trained MF/BPR-AUC embeddings** as one more factor model. Relevance: **low-med**.

---
## SHORT ENTRIES (6-14)

### 6. Avito Context Ad Clicks (2015)
- Comp: https://kaggle.com/competitions/avito-context-ad-clicks/overview [200]. Ads shown in a search context, user history, logloss.
- Code: https://github.com/owenzhang/kaggle-avito [200] (README states "Winning solution to the Avito CTR competition"); https://github.com/Gzsiceberg/kaggle-avito [200] (another top solution).
- Idea: **aggregate/rank features over all ads in the same search/slate and over the user's history of the same ad** (we have). Relevance: **low** for us.

### 7. Huawei DIGIX CTR 2020 (same organiser)
- Comp page / winners: https://developer.huawei.com/consumer/cn/activity/digixActivity/digixWinnersDetail/101655281685926449 [200]
- 1st place repo (README read): https://github.com/digix2020/digix2020_ctr_rank1 [200] ; blog (CN): https://hub.baai.ac.cn/view/4496 [200] ; https://cloud.tencent.com/developer/article/2146686 [200]
- Read facts (README): LightGBM-only; three variants: **full-feature 0.813, sliding-window 0.811, no-unique-feature 0.811, fused 0.814** on final B. AUC range ~0.81-0.82, same as ours.
- Idea: **diversity by data/feature view rather than by model family** (train on sliding windows of cutoffs, drop the highest-cardinality/ID features) and fuse. Relevance: **med** (ensemble +0.001-0.003 typical; we already blend LGBM+MLP but both on the same view).

### 8. Expedia ICDM 2013 (hotel ranking)
- Comp: https://www.kaggle.com/competitions/expedia-personalized-sort/overview [200] ; winners https://www.kaggle.com/competitions/expedia-personalized-sort/overview/winners [200, rate-limited when fetched] ; a group write-up https://chihchiehlin.com/documents/DMT_2_group55.pdf [200]
- Ideas: negative-row downsampling and per-search normalisation of features (have). Relevance: **low**.

### 9. OTTO (Kaggle 2022-23), pure IDs
- Comp: https://www.kaggle.com/competitions/otto-recommender-system/overview [200]
- 1st place (read): https://www.kaggle.com/competitions/otto-recommender-system/writeups/mrkmakr-1st-place-solution [200] ; Chris Deotte code: https://github.com/cdeotte/Kaggle-OTTO-Comp [200]
- Read facts: candidate generation + LGBM/NN rerank; NN trained with **multiple future items as positives** and shared item embedding.
- Idea: **multi-positive "set" training for the user encoder**: since the taste is static and order carries no information in our data, train a bag-of-history autoencoder (predict held-out clicks from the rest, many positives per user) rather than next-item sequence models. Relevance: **med** (matches the planted structure; co-visitation tried and flat).

### 10. H&M (Kaggle 2022)
- Comp: https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/overview [200]
- 1st place (read): https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/writeups/senkin13-30crmnsia-1st-place-solution [200] ; CN summary https://lukan217.github.io/2022/05/14/kaggle%20HM%E6%8E%A8%E8%8D%90%E8%B5%9B%E8%8E%B7%E5%A5%96%E6%96%B9%E6%A1%88%E6%80%BB%E7%BB%93/ [200] ; paper https://arxiv.org/pdf/2210.08473 [200]
- Idea: **graph-propagated embeddings (ProNE / spectral propagation) on the user-item click graph** as another embedding family beyond PPMI-SVD. ProNE: https://www.ijcai.org/proceedings/2019/594 [200]; code: https://github.com/shenweichen/GraphEmbedding [200]. Relevance: **low-med** (higher-order neighbours may denoise k=12 factors; H&M saw small positive gains, not verified on our structure).

### 11. TalkingData AdTracking Fraud (AUC)
- Comp: https://www.kaggle.com/competitions/talkingdata-adtracking-fraud-detection/overview [200]
- 1st place (read): https://www.kaggle.com/competitions/talkingdata-adtracking-fraud-detection/writeups/flowlight-komaki-shuffle-1st-place-solution [200] ; code https://github.com/flowlight0/talkingdata-adtracking-fraud-detection [200]
- Read facts: negative downsampling with bagging; **LDA/NMF/PCA embeddings of categorical-combination "documents"** as features; rank-weighted blend.
- Idea: **fit LDA / NMF (non-negative factor models) with k about 12 on the user x item click counts**, and use dot(theta_user, phi_item). If the generator's tastes are non-negative/simplex-like, LDA/NMF recovers them better than SVD (data_dig_v1 says the k=12 fit quality is the bottleneck). Relevance: **med-high** (cheap, sklearn, directly targets the weakest part of our pipeline).

### 12. RecSys Challenge 2015 (YooChoose)
- https://github.com/romovpa/ydf-recsys2015-challenge/ [200] ; paper DOI https://doi.org/10.1145/2813448.2813510 [403*]
- Two-stage (buy session, then items): this is our event-level decomposition (flat). Relevance: **low**.

### 13. Avazu / Criteo
- Avazu: https://www.kaggle.com/competitions/avazu-ctr-prediction/writeups/4-idiots-4-idiots-solution-libffm [200, read] ; https://github.com/ycjuan/kaggle-avazu [200]
- Criteo: https://www.kaggle.com/competitions/criteo-display-ad-challenge/writeups/3-idiots-3-idiots-solution-libffm [200] ; https://github.com/ycjuan/kaggle-2014-criteo [200]
- Read facts: **FFM** (field-aware FM) with per-field-pair latent vectors, best k about 8; last-day validation; Criteo adds GBDT leaf features into FFM (the origin of the leaf trick above). FFM slides: https://www.csie.ntu.edu.tw/~r01922136/slides/ffm.pdf [200].
- Idea: **FFM/FM with fields {user, candidate, history-item bag}**. Relevance: **low** (the two-tower/PPMI machinery is the same bilinear form; FFM adds field-specific factors that a random-ID synthetic generator probably does not have).

### 14. Coveo 2021 / Dressipi 2022 / ShareChat 2023 / Tencent
- Coveo (read paper): https://sigir-ecom.github.io/ecom21DCPapers/paper7.pdf [200] ; 1st place code https://github.com/DeepBlueAI/SIGIR_eCOM_2021_Data_Challenge_1st_Place [200] ; NVIDIA Merlin write-up https://medium.com/nvidia-merlin/winning-the-sigir-ecommerce-challenge-on-session-based-recommendation-with-transformers-v2-793f6fac2994 [403*]
- RecSys 2022 (Dressipi): https://www.recsyschallenge.com/2022/index.html [200] ; paper https://dl.acm.org/doi/pdf/10.1145/3523227.3552534 [403*]. (snippet) Swing/graph similarity features + LGBM.
- RecSys 2023 (ShareChat): Layer 6 https://layer6.ai/publications/recsys-challenge-2023-1st-place-robust-user-engagement-modeling-with-transformers-and-self-supervision/ [200] ; DOIs https://doi.org/10.1145/3626221.3627285 and https://doi.org/10.1145/3626221.3627289 [403*]. Contrastive self-supervision for a tabular transformer; relevance **low**.
- Tencent Social Ads 2018: https://github.com/DiligentPanda/Tencent_Ads_Algo_2018 [200] ; https://github.com/totoruo/TSA2018-rank12 [200]. Tencent 2020 (demographics) not a CTR analogue, skipped. Relevance **low**.
- DIGIX 2021/2022: I found no verified write-ups beyond the 2020 champion; not included.

---
## OVERALL TOP-5 UNTRIED IDEAS (ranked by expected value / cost)

1. **LDA / NMF (k about 12) user-item factors** (TalkingData embeddings; targets "k=12 fit quality is the bottleneck"). Relevance **med-high**. Cost: low. Source: TalkingData 1st place writeup (#11).
2. **Global cross-event pairwise-AUC objective** (RankNet-style on sampled positive-negative pairs from different events, or XGB `rank:pairwise` over random mixed groups) as an extra blend member. Relevance **med**. KDD Cup 2012 NTU (#5), Outbrain 3rd (#2).
3. **GBDT-leaf -> SVD/embedding -> MLP/FM member** (+0.3 MRR points in Layer 6's blend; +0.0012 MAP ensemble in Outbrain 3rd). Relevance **med**. Needs OOF-safe leaves.
4. **Diversity by data view + recency weighting**: sliding-window / no-high-cardinality-feature models fused (DIGIX 2020: 0.813/0.811/0.811 -> 0.814), plus exp-decay sample weights or a "stale/future" model (Outbrain 3rd: +0.001-0.002). Relevance **med**.
5. **Blend-space and shift audit (cheap, safety)**: compare probability / logit / rank averaging for NN bags and ablate recency/popularity-derived features on a stale-matched fold (MIND: val-up features hurt a week ahead). Relevance **med/high as a guard**. Runner-up: multi-positive bag-of-history autoencoder (OTTO) and ProNE propagation (H&M), both low-med.

## Caveats
- Most gains quoted are on other metrics/datasets; none is guaranteed to carry to a synthetic, ID-only, global-AUC setting.
- The Kaggle pages are JS apps; "200" confirms the URL resolves, and the write-ups marked "read" were retrieved with WebFetch.
- Search-summary-only claims are labelled (snippet).
