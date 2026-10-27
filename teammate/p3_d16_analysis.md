# Teammate submission `submission_p3_d16` — what the predictions reveal

_Analysis on 26 Oct 2026 (Europe/Dublin, UTC+0). Inputs: `submission_p3_d16.zip` (read-only), `v2work/out/submission_v8.csv`, `v3/data` (train events Jan 20–26, eval events Jan 27–Feb 2).
Work files: `teammate/work/` (scripts `feats.py`, `a1.py`–`a6.py`, feature table `X2.parquet`, `segments.md`). Nothing was uploaded._

There are no labels, so nothing here measures AUC. Everything below is inferred from the predictions alone, and each claim carries a confidence level.

## 1. Basic checks
| check | p3_d16 | v8 |
|---|---|---|
| rows / unique rowids | 529,390 / 529,390, none missing or extra | same |
| row order | identical to the eval file | identical |
| mean prediction | **0.0604** (train CTR is 0.0629) | 0.0941 (inflated by negfrac 0.5) |
| implied clicks per event (sum of the 10) | 0.60 (train: 0.63) | 0.94 |
| range | 0.000327 – 0.6876 | 0.00056 – 0.7827 |
| median / p99 | 0.0379 / 0.366 | 0.0514 / 0.524 |
| distinct values | 162,211 (6 decimals) | 528,017 |
| ties within an event | 38 rows | 0 |
| file format | CRLF line endings, 6 fixed decimals, zip entry `create_system=0` (DOS/Windows), entry time 18:26:52 local wall-clock | LF |

- **Ties come from rounding, not from a coarse model (high confidence).** If v8 is shifted to the same mean and rounded to 6 decimals, it gives about 170k distinct values and a maximum tie of 34. p3 has 162k distinct values and a maximum tie of 24. Its value grid is 99.9% occupied in [0.01, 0.02]. Only 78 rows share an (item, value) pair. So the score is continuous, which rules out a small tree or an item lookup.
- **The output is calibrated to the base rate (high confidence).** Over the 10 rows of an event, the predictions sum to 0.60 on average, against 0.63 clicks per event in train. That points to a pointwise logloss model trained on all negatives, or one corrected for subsampling.
- **It is not a per-event softmax or listwise-normalised score (high confidence).** Per-event sums range from 0.30 (p1) to 0.95 (p99).
- **The file was probably written on Windows (medium confidence).** Evidence: CRLF line endings, the zip host flag, and probably pandas `to_csv(float_format='%.6f')`. The zip time was 18:26 local wall-clock, but the file reached the box at 17:18 UTC. So their clock runs at least about 1 h ahead of UTC, either a UTC+1 or more easterly timezone or a wrong clock (low confidence).

## 2. Comparison with v8
- Pearson 0.855, Spearman 0.851, logit Pearson 0.863.
- Mean within-event rank correlation: 0.797. Same top-1 item in 64.7% of events. Correlation of event-level sums: 0.779.
- Logit fit: logit(p3) = 0.706·logit(v8) − 1.18. p3 is flatter (less confident) than v8. Logit sd is 1.22 for p3 versus 1.49 for v8.
- v8 alone explains R² 0.75 of logit(p3) on held-out events; adding the simple features below raises it to 0.76.

Decile agreement (row = p3 decile, column = v8 decile, % of row):
```
 [[65 24  8  2  1  0  0  0  0  0]
 [22 33 23 13  6  2  1  0  0  0]
 [ 8 21 25 21 13  7  4  1  0  0]
 [ 3 12 19 21 19 13  8  4  1  0]
 [ 1  6 12 17 21 19 13  8  3  0]
 [ 0  3  7 12 17 22 20 13  6  1]
 [ 0  1  4  8 13 18 22 20 12  3]
 [ 0  1  2  4  8 12 19 25 22  7]
 [ 0  0  1  1  3  6 12 23 34 20]
 [ 0  0  0  0  0  1  2  7 21 69]]
```
The top and bottom deciles agree well (65–69% on the diagonal). The middle deciles agree much less (21–25%).

**Where the two disagree most** (full tables in `work/segments.md`):
- **Cold users (empty history, 6,650 rows, 665 events).** Spearman with v8 is only 0.65, and the within-event rank correlation is just **0.21** (0.80 for warm users).
  - v8 ranks cold users' candidates using item popularity in train histories (coefficient +0.33).
  - p3 ignores train popularity (−0.05). It also ignores item CTR: within-event rank correlation is −0.03 for p3 versus +0.26 for v8.
  - In p3, cold events push the **last candidate (position 9)** to a mean rank of 8.6 out of 10. Train cold users show no such position effect (CTR is flat by position), and warm users don't show it either.
  - This looks like a bug or artifact in their empty-history handling, for example padding the history with something tied to the last candidate (medium confidence that it is an artifact; the cause is unknown).
- **Users with no train events (14,530 rows):** p3 ranks them lower than v8 does (mean rank 0.561 vs 0.589).
- **Item already in the user's history (13,467 rows):** both models rank these near the top (p3 0.960, v8 0.941).
- **Not different:** the ratio of means is a constant 0.64 across hour bucket, day type, eval day (0–6), item popularity, history length and recency. Neither model shows any time drift.
- **The 1% largest rank disagreements** sit in slightly longer histories, more recent users, and items with no history match (`hin`≈0). In 60% of them, p3 is the higher one.

## 3. What signals p3 uses (surrogate models on eval-row features)
The target is logit(p3), or logit(v8) as a control. The surrogate is LightGBM with depth ≤ 6 (or a linear model), fit on 80% of events and scored on the other 20% (split by event).

| features | R² p3 | R² v8 |
|---|---|---|
| item train CTR only | 0.001 | 0.001 |
| simple train-only features (21: history length, cold flags, time bucket, day type, day, hour, position, item popularity and CTR, user CTR and counts, recency, item-in-own-history, item frequency in train histories, within-event ranks), linear | 0.108 | 0.086 |
| same, tree depth 3 / depth 6 | 0.124 / 0.122 | 0.098 / 0.097 |
| + history-similarity features (train co-click sum, cosine of candidate vs history centroid on train-only SVD k = 8/16/32/64) | **0.492** | **0.450** |
| simple + one SVD cosine: k8 / k16 / k32 / k64 | 0.279 / 0.357 / **0.377** / 0.366 | 0.260 / **0.327** / 0.323 / 0.309 |

Feature gain shares:
- **Simple features only:** item-in-own-history 0.70, user CTR 0.09, user train rows 0.07, then item popularity and CTR at about 0.03 each.
- **With history features:** svd32 cosine 0.33, svd64 0.19, svd16 0.15, item-in-own-history 0.08.

What this says:
- **p3 relies on the user's click history, and on candidate–history similarity in an item-embedding space (high confidence).** Within warm events, its rank correlation with the svd32 cosine is 0.49 (v8: 0.47), and with the co-click sum 0.28 (v8: 0.26).
- **Item and user CTR priors play a small role (high confidence).**
- **p3 leans on history similarity at least as much as v8 does.** It is slightly more "pure history" than v8.

## 4. Leak check (eval-row aggregates)
What can and cannot leak:
- **User-level leak routes (future events of the same user, user event counts in eval) cannot exist in this data (certain).** Every eval user has exactly one eval event (52,939 users, 52,939 events), and no eval history timestamp reaches the eval period (the latest is 1 minute before Jan 27 00:00 UTC). So there was nothing to test there.
- **The aggregates that can exist are item-level.** I tested four:
  - `ev_icand`: how many eval rows show the item;
  - `ev_icand_day`: the same count, for the same eval day;
  - `ev_ifut`: later eval rows that show the item;
  - `ev_ihist`: how many eval users have the item in their history.

Results:
- **No overall excess.** Adding the four eval aggregates (plus their within-event rank) to the train-only surrogate raises R² by **+0.0013 for p3** against **+0.0018 for v8**, which is known to be clean. That is no excess at all.
- **Warm rows: no sign.** Within-event regression coefficients on `ev_icand` are −0.006 for p3 and −0.004 for v8.
- **Cold rows: a soft sign.** I controlled for train item exposure over all of train, the last 2 days, the last 1 day and the last 6 hours, plus frequency in train histories and position. Within-event rank-regression coefficients (95% CIs from bootstrapping over events):
  - p3: `ev_icand` **+0.22 [+0.18, +0.26]**;
  - v8: `ev_icand` +0.05 [+0.01, +0.09].

  So for users with no history, p3 follows how often the item is shown *during the eval week* much more than v8 does, and more than any train-only exposure I could build explains. `ev_icand` is strongly collinear with train exposure (r ≈ 0.81–0.85), so the residual is a small slice of variance.
- **How to read this:** it fits a frequency/count encoding computed on train+eval combined, which is a common habit (e.g. `pd.concat([train, eval])['item'].value_counts()`). It could also be a legitimate popularity feature built differently from mine, such as a decayed count or a different train window.
- **Confidence that p3 uses an eval-row aggregate: low–medium (about 40%).** It is only visible on cold rows, which are 1.3% of rows, so the impact on AUC would be small either way.
- **Recommendation: ask the teammate directly** whether any count, frequency or target encoding, or any embedding/SVD fit, used evaluation rows.

## 5. Most likely model and training (with confidence)
| inference | confidence |
|---|---|
| Continuous-score model, not a small tree or lookup; output rounded to 6 decimals | high |
| Pointwise binary logloss trained on (nearly) all negatives, or with calibration; not per-event normalised | high |
| Uses the user's history sequence heavily, through candidate–history similarity in an item-embedding / co-click space, much like v8's history features | high |
| Weak item-side priors (item CTR almost unused); cold users handled poorly, with a position-9 artifact | medium–high |
| No time-drift handling or time features that matter (flat across eval days, hours, day type) | medium |
| Model family (GBDT on embedding features vs neural sequence/two-tower): undecided. The flat item-CTR use and the cold-path artifact slightly favour a neural or embedding-similarity model | low |
| "d16" = embedding dimension 16? The svd16 cosine explains p3 slightly *less* than svd32/64 (0.357 vs 0.377), so the predictions don't support "dim 16" particularly; it could equally be depth 16 or 16 days. Not determinable | low |
| Written on Windows with pandas, in a timezone about UTC+1 or further east | medium / low |

## 6. How different it is, and the blend probe
- **Rank diversity vs v8:** Spearman 0.85. That is a real second opinion, mostly on mid-ranked rows and cold users.
- **Rank-average blends**, compared with v8:

| w (p3) | Spearman vs v8 | mean absolute rank shift |
|---|---|---|
| 0.1 | 0.9986 | 0.012 |
| 0.2 | 0.9942 | 0.023 |
| **0.3** | 0.9867 | 0.035 |
| 0.5 | 0.9617 | 0.059 |

- **Probe file written:** `teammate/probe_blend_p3.csv.zip`, i.e. `0.7·rank(v8) + 0.3·rank(p3)` with the header `rowid,predicted_probability`, 529,390 rows in eval order, 8 decimals, values in (0, 1). `v3/validate_submission.py` reports it VALID.
  - csv sha256 `06a367e6f8acecd23350696cd26a0992000c8ffbae55887e322276d07dc82880`
  - zip sha256 `9edbd82c431527caacce3476bf4c405bba5f455a42662c04382683d64fcb295e`
- **Not uploaded.**
- **Is it worth one submission?** Yes, as a probe, if the teammate's public score is within about 0.01 of v8 (≥ about 0.805). If p3 is weaker than that, try w = 0.2 instead.
- **Caveat:** if the cold-row eval-frequency signal turns out to be a real eval aggregate, the blend inherits a little of it. Clear that with the teammate first, given the no-eval-aggregate rule.
