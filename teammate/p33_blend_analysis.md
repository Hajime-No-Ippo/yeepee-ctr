# Teammate blend `submission_p33_blend_w70` (public 0.815): what changed and why it helps v8

_27 Oct 2026, Europe/Dublin (UTC+0). Input `teammate/submission_p33_blend_w70.zip` (read-only; predictions only).
Work dir `teammate/work33/` (scripts a1–a7, sim.py, shift.py, boot.py; tables `segments33.md`, `sim_results.json`; recovered p33 `p33_recovered_ranks.parquet`).
Proxy model run: `v9/ens/p3like_b_c{25,23}_s0.npy` (log `v9/logs/p3like.log`). Nothing uploaded, pushed or deleted._

## TL;DR
- **The recipe is solved exactly (certain):** `0.7·rank_pct(v8) + 0.3·rank_pct(p33)`, using global average ranks over all 529,390 rows (pandas `rank(pct=True)`), written to 8 decimals. Rebuilding it from our v8 file plus the recovered p33 ranks matches the teammate's file bit for bit on 100% of rows. So they used **our v8 file unchanged**.
- **p33 is not p3_d16.** It is a new or retrained model that sits much closer to v8. Spearman with v8 is 0.929 (p3_d16: 0.851), and the two share the same top-1 item in 74% of events (p3_d16: 65%).
- **Cold and short-history users:** p33 no longer relies on an item prior for cold users. Its cold ranking is close to flat. The last-candidate bias is mostly gone, and the cold leak sign is mostly gone too.
- **Stale rows:** p33 does not hold up better than v8 there. I see no sign of decay or recency features.
- **Why the blend helps:** the movement is spread across all segments, roughly in proportion to their row counts. It is not a fix for one of v8's weak spots. My validation proxy measured this:
  - our own "p3-like" model, built from a subset of v8's features, makes v8 **worse** when blended at 0.3 (−0.0009 at C=25, −0.0013 at C=23);
  - moving whole segments up or down can't produce the gain either (≤ +0.00003).
  - So the +0.0006 must come from p33 carrying **information v8 doesn't have**, most likely a different representation of the history (a sequence or embedding model). It is not segment handling.
- **Caveat:** +0.0006 is small. In validation, the paired bootstrap SD of a blend-vs-v8 difference is about 0.00013 on a full eval-sized set and about 0.0004 on 30% of it. If the public board covers only part of eval, the gain is real-ish but weak (roughly 1.5–4 SD depending on the public fraction).

## 1. Blend recipe and the recovered p33 (certain)
- Computing `k = (b − 0.7·rank(v8)/N)/0.3·N` gives integers or half-integers on every row. The 7,330 half-integers are p33 ties, consistent with 6-decimal rounding as in p3_d16.
- The values run from 1 to N, and their sum equals N(N+1)/2 exactly. That makes k a complete average-rank vector.
- Other weights (0.65, 0.75, 0.8) and other rank conventions (r/(N+1), (r−1)/(N−1)) all fail this integer test (≤ 49% of rows fit, against 100%).
- The rank is global, not per-event. Logit or probability averaging is ruled out.
- File details: CRLF line endings, zip entry written by a DOS/Windows host, entry time 16:16 on the teammate's clock. That is the same toolchain as p3_d16.
- Only p33's **ranks** are recoverable, not its probabilities. That is all AUC needs.

## 2. How p33 differs from p3_d16
| | v8 | p3_d16 | p33 |
|---|---|---|---|
| Spearman vs v8 | – | 0.851 | **0.929** |
| within-event rank corr vs v8 / same top-1 | – | 0.797 / 64.7% | 0.885 / 74.3% |
| Spearman vs p3_d16 | 0.851 | – | 0.894 |
| best linear fit in normal scores | | | 0.59·v8 + 0.40·p3, R² 0.912 |
| tree fit p33 ~ f(v8, p3) | | | R² 0.913; adding 29 train-only features: 0.920 |

- **p33 is mostly what you'd get by mixing v8-like and p3-like signal (high confidence).** About 8% of its normal-score variance is its own.
- That residual is barely explained by simple or history-similarity features (R² 0.06; the top gains are day type, svd64 and history length).
- So the extra part is either a different representation or noise. I can't tell which from the predictions alone.
- **Did they borrow v8-style features? Medium confidence, yes, or something equivalent.** p33 tracks v8 far more closely than p3 did, especially within events.
- An alternative I can't exclude (about 25%): p33 was itself trained or stacked on v8-like outputs.

### 2a. No-history users (6,650 rows / 665 events) — Eric's focus
Within-event rank correlation with item signals, cold events only:

| | train popularity `it_n` | popularity in train histories | last-day exposure | item CTR | eval-week exposure |
|---|---|---|---|---|---|
| v8 | **0.51** | **0.55** | 0.49 | 0.26 | 0.47 |
| p3_d16 | 0.28 | 0.26 | 0.29 | −0.03 | 0.31 |
| p33 | **0.06** | 0.08 | 0.06 | 0.08 | 0.08 |

**v8 is the only model with a real cold fallback (high confidence).** It ranks the candidates for cold users by item popularity.

**p3_d16 used a fixed per-item score (high confidence).** The same item gets almost the same within-event rank in every cold event: rank variance 1.85 against 8.25 if random.

**p33 is close to random for cold users (high confidence).** Its rank variance is 4.7, and its correlation with popularity is about 0. It probably zeroes or pads the empty history and ranks by weak residual item effects.

Cold agreement with v8 is low for both: Spearman 0.445 for p33 and 0.644 for p3_d16; within-event correlation 0.18 and 0.21.

**Last-candidate bias is mostly fixed (high confidence):**

| | mean within-event rank of position 9, cold (1–10) | share of cold top-1 picks at position 9 |
|---|---|---|
| v8 | 5.51 | 10.5% |
| p3_d16 | **8.58** | **31.4%** |
| p33 | 6.15 | 15.5% |

- Position 1 is also 6.15 for p33, and the controlled regression coefficient on position is −0.004. What's left looks like noise from a near-flat score, not a systematic bias.
- Warm users show no position effect in any of the three models.

**Leak sign mostly gone (medium confidence).** This is the cold-row within-event regression on eval-week exposure `ev_icand`, controlling for all train exposure windows:

| | coefficient [95% CI] |
|---|---|
| p3_d16 | +0.220 [+0.18, +0.26] |
| p33 | **+0.075 [+0.03, +0.13]** |
| v8 (clean) | +0.046 [+0.01, +0.09] |

- p33 is now statistically close to v8. Warm rows show nothing for any model.
- Either the teammate dropped the train+eval frequency encoding, or p33's cold scores are simply too flat to show it.

**Short history (1–2 items, 24,890 rows):**
- Within events, p33 ranks on own-history hits and co-click or SVD similarity (0.53 / 0.40 / 0.30), the same way v8 does (0.53 / 0.44 / 0.32). It uses no popularity (−0.01 against v8's +0.02).
- At event level, p33 puts these users lower overall (mean rank 0.521 against v8's 0.557). That is the largest segment-level gap anywhere.
- **Validation says this placement doesn't matter.** Shifting v8's short-history rows by ±0.01–0.04 changes AUC by at most +0.00001 (C=25, C=23).

**What this means for no-history users (high confidence):**
- They are 1.3% of rows, and the blend barely moves them (0.5% of total squared movement), although it flips the top-1 item in 30% of cold events.
- On validation, cold-segment AUC is about 0.51–0.56 for every model. In global AUC they are worth about 0.00002.
- **The +0.0006 does not come from the no-history logic.**

### 2b. Stale rows (Data Wrangler buckets) — Eric's focus
| staleness | rows | Spearman v8–p33 | within-event v8–p33 | mean rank v8 / p33 |
|---|---|---|---|---|
| 0–24h | 66.6% | 0.935 | 0.899 | 0.490 / 0.486 |
| 24–48h | 16.6% | 0.927 | 0.893 | 0.513 / 0.522 |
| 48h+ | 16.8% | 0.903 | 0.875 | 0.528 / 0.535 |

- **Disagreement grows with staleness, for p33 and p3_d16 alike** (p3: 0.857 → 0.824). That is expected: every model gets noisier there.
- **No sign that p33 holds quality better on stale rows (medium confidence):**
  - its within-event use of the signals that stay stable as rows age (own-history hit 0.50 → 0.52, svd 0.45 → 0.43, co-click 0.25 → 0.27) tracks v8's almost exactly, just slightly weaker on svd;
  - its use of fast-decaying recent exposure (last 6h or 1 day) is about 0 in every bucket, as it is for v8.
- **No evidence of decay or time-since-last-activity features (medium confidence):**
  - p33's event-level shift against v8 correlates only weakly with hours since cutoff (+0.08) and not at all with time since the user's last history item (−0.02);
  - all three models rank stale and long-gap users' events higher at a similar strength (Spearman with staleness: v8 0.23, p33 0.26);
  - shifting v8's 24h+ or 48h+ rows up or down on validation never helps (best +0.00003). v8's placement of stale events is already about right, so p33's slight upward push there is neutral.

## 3. Where the blend moves v8, and why it adds
**The blend moves v8 within events, not between them.**
- Of the total squared rank movement (blend − v8), **85% is within-event re-ordering and 15% is event-level shifts.**
- The top-1 item changes in 8.4% of events overall: 8.1% at 0–24h, 8.8% at 24–48h, 9.4% at 48h+, 9.1% for 1–2 history items, and 29.9% for cold users.

**Share of the movement by segment, against share of rows** (full tables in `work33/segments33.md`):

| segment | rows | share of movement | event-level share |
|---|---|---|---|
| 0–24h | 66.6% | 64.9% | 59.8% |
| 24–48h | 16.6% | 16.2% | 16.2% |
| 48h+ | 16.8% | **18.8%** | **24.0%** |
| history 0 | 1.3% | 0.5% | 0.9% |
| history 1–2 | 4.7% | **6.1%** | **9.2%** |
| history 3–5 | 11.7% | 12.4% | 13.7% |
| history 6–10 | 19.4% | 19.5% | 20.3% |
| history 11–20 | 62.9% | 61.4% | 55.9% |
| item popularity, time of day, position | | proportional to rows | |

The movement is only slightly over-weighted on v8's weak spots (48h+ and short history), by a factor of about 1.1–1.3.

### Measured proxy (C=25 primary, C=23 with 48h+ rows)
Same recipe, 0.7·rank(v8 proxy) + 0.3·rank(partner), where the v8 proxy is our trim z8:
- **p3like:** our out-of-fold LightGBM on only v8's history, co-click, SVD, MF and LR-similarity features (265 features), with no item or user priors, no taste features and no x_clctr/x_climp. Alone it scores 0.7997 / 0.7742, with Spearman 0.939 / 0.924 against the proxy, which is close to p33's 0.929.
- **noTT:** v8 without the taste features.

| partner, w = 0.3 | Δ AUC C=25 | Δ AUC C=23 | 24–48h | 48h+ | history 0 | history 1–2 | history 11+ |
|---|---|---|---|---|---|---|---|
| p3like | −0.0009 | −0.0013 | −0.0013 / −0.0015 | −0.0013 | +0.004 / +0.002 | +0.0012 / +0.0010 | −0.0009 / −0.0013 |
| noTT | −0.0003 | −0.0009 | −0.0006 / −0.0011 | −0.0009 | +0.001 / −0.002 | +0.0007 / +0.0012 | −0.0003 / −0.0011 |

- At w = 0.1, p3like gives 0.0000 / −0.0001.
- Applying the blend only to one segment, at C=25 / C=23:
  - 24h+ only: −0.0003 / −0.0004;
  - history ≤ 5 only: −0.00004 / −0.0001;
  - cold only: +0.00001 / +0.00002.
- Breaking the change into event-level and within-event parts: both parts hurt for p3like (within-event −0.0011 / −0.0016; event-level −0.0002).

**What this shows, with confidence levels:**
1. **A model built from the same features as v8 does not add anything, however it is tuned (high confidence).** Blending only helps on very short histories and cold users, and those segments are too small to matter. Stale rows do not benefit at all.
2. **The teammate's blend did add +0.0006.** So p33 must carry signal that is independent of v8's feature set, mostly in the within-event ordering of warm users across all segments (medium–high confidence).
   - The likely sources are a learned sequence or embedding representation of the history, or a different training-data construction. p3's history-embedding similarity was already its strongest trait.
   - It is not cold-user handling, not staleness handling and not time features.
3. **p33 alone is probably close to v8, around 0.80–0.81 public (low–medium confidence).**
   - In our simulations, a partner with Spearman 0.93 that is 0.011–0.016 weaker than v8 costs about 0.001 at w = 0.3.
   - A gain of +0.0006 at w = 0.3 therefore needs a partner that is either much more complementary or only slightly weaker.
   - p3_d16 at 0.78 would most likely have hurt at w = 0.3, which matches the earlier advice to use w = 0.2 or less for it.
   - It would help to ask the teammate for p33's own public score.

## 4. Probe (optional)
I did not write a new submission.
- The recovered ranks (`work33/p33_recovered_ranks.parquet`) are enough to rebuild any v8/p33 rank mix exactly.
- I have no p33 validation predictions, so any other weight would be an unvalidated guess.
- If one probe is wanted: w = 0.4 tells you whether p33 is strong (a gain) or the 0.3 result was mostly noise (a drop), but it risks about −0.001.
- The more useful step is to ask the teammate for p33's out-of-fold validation predictions. With those we can tune the weight on C=25/C=23 properly.

## Questions worth asking the teammate
1. What is p33's architecture, and is it a sequence model over the history?
2. Did any frequency, count or target encoding, or any embedding fit, use eval rows? The leak sign is weaker now but not zero.
3. What is p33's standalone public score?
4. Can they share out-of-fold predictions for a week-25 (Jan 25) or week-23 holdout?
