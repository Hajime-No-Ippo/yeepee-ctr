# Data dig v2: the rank-12 latent structure (written 17:47 IST, 23 Oct 2026)

Setup: same sample as v1 (D.parquet: 63,220 val rows, 5,612 events; val25 2-day window plus val26 1-day window; 710 J26 events appear in both cuts).
Item vectors and models are fitted on train events with ts < J25 (all users, train only). The same J25 model scores both cuts, so v26 is 1 day stale and its numbers are conservative.
Metrics: global AUC, within-event AUC, and resid = the 5-fold user-hash stacking test on the base logit (7 leaves, 150 rounds, 2–3 seeds). Noise is about ±0.0015.
Nothing is aggregated over eval rows. No pipeline files were touched. Everything ran at ≤2 threads under nice 19 (a 3-thread screen.py was running alongside).
Scripts are in data_dig_v2_scripts/ (lat.py is the shared loader).

## 1(a) The click-window caveat is explained: missing row normalisation, not drift
- The matrix had column scaling 1/sqrt(pop+5) but no row scaling. Clicks per user are heavy-tailed: mean 17, max 237 pre-J25; in-window median 4, p99 39, max 217.
  With beta = 0, heavy users' rows produce "spike" singular vectors that out-rank the 12 planted factors on the flat spectrum.
  last-N capping happened to cap row length, which is why it worked.
- Adding row scaling 1/len^beta (a1.py, a2.py; within-AUC 25/26):

  | clicks used | beta 0 | beta 0.5 | beta 1 |
  |---|---|---|---|
  | all pre-C clicks | 0.625/0.634 | **0.782/0.781** | 0.524/0.513 |
  | since J13 | 0.586/0.603 | 0.763/0.771 | — |
  | last-30 | 0.774/0.767 | 0.778/0.776 | — |
  | window only | 0.516/0.525 | 0.546/0.552 | — |

  Window-only stays bad because there is too little co-occurrence (median 4 clicks per user), not because taste changes.
  The best setting is alpha 0.5, beta 0.5, all clicks, k13 (drop comp 1, keep comps 2..13 scaled by s). Neighbouring settings (k12 vs 13, alpha 0.3) move it by about ±0.004, so exact SVD on this spectrum is fragile.
- No drift. For 11,135 users with ≥8 warm and ≥8 window clicks, centroid cosines in V:
  - same user, warm vs window: 0.851
  - warm early half vs warm late half: 0.795
  - random halves: 0.807
  - random users: −0.004

  User vectors are fixed over time; the generator does not re-draw them.
- New SVD feature (a=.5, b=.5, all clicks, k13, history-centroid cosine): within 0.782/0.781, resid **+0.0130/+0.0100**. On this sample, v1 V13 gives +0.0096/+0.0066.

## 1(b) Model comparison at d≈12 (within-AUC 25/26; resid)
| model | within | resid |
|---|---|---|
| implicit ALS d12 alpha 20, lambda 1, 10 iters, item-cos centroid | 0.766/0.764 | +0.0093/+0.0069 |
| ALS d8 / d16 | 0.69–0.70 / 0.763/0.756 | — |
| SVD a.5 b.5 k13 | 0.782/0.781 | +0.0130/+0.0100 |
| **two-tower softmax-within-event, SVD-initialised, tied, no item bias** | **0.797/0.800** | **+0.0194/+0.0140** |
| same, with outside option and null head | 0.794/0.794 | +0.0197/+0.0140 |
| same, plus free per-user vector | 0.792/0.790 | +0.0177/+0.0130 |
| two-tower from random init | 0.54 | fails |
| two-tower untied | 0.778/0.771 | — |
| two-tower d8 / d15 | 0.746/0.741 / 0.794/0.797 | — |

- Seed 1 reproduces seed 0 (0.7975/0.7994).
- Learned item biases hurt: they memorise. With frozen vectors plus biases, within drops to 0.63. Item bias carries no transferable signal, consistent with v1.
- BPR and logistic MF were not run. The softmax-within-event model is the discriminative version of both.

## 1(c) The generator looks like a conditional logit on u·v
- On clicked val events, softmax(2.0 · dot) over the slate is well calibrated: predicted vs observed by decile 0.006/0.008 … 0.192/0.205, 0.392/0.385. NLL is 1.77 vs 2.30 for uniform.
- Outside option: slate logsumexp alone predicts whether the event gets any click (event AUC 0.603/0.606). This is consistent with P(no click) = exp(c)/(exp(c)+Σexp(u·v)).
- The learned null head (c0 + w·u) does not do better than raw lse.
- The two-tower fit is effectively the direct estimate of the generator. Its AUC is in the 1(b) table.

## 1(d) Per-user ridge on fixed item vectors is worse than the history centroid
- Per-user ridge (fixed SVD vectors, the user's pre-C impressions, clicked minus mean, lambda 10): within 0.738/0.746, vs the centroid at 0.782/0.781.
- Stacking both gives +0.0142/+0.0113 vs +0.0130/+0.0100 for the centroid alone. That is within noise to +0.001.
- The free per-user embedding in the two-tower model also hurt (overfitting).

## 2. Event level: the 0.62 cap breaks
Event-click AUC from a 2-fold logistic regression over val events:

| inputs | AUC 25/26 |
|---|---|
| user CTR (shrunk) | 0.610/0.595 |
| + \|u\| + history length (user-level) | 0.609/0.597 |
| + slate lse + slate mean of u·v | **0.653/0.644** |
| all of the above | **0.664/0.658** |

The gain is slate-specific, not user-level. Max is weaker than lse or mean (0.584 alone).
Row-level, adding the event aggregates (lse, mean, dot − mean) to the tower dot raises resid from +0.0194/+0.0140 to **+0.0220/+0.0156**.
Adding the SVD cosine and v1 V13 as well gives +0.0217/+0.0160, so the SVD features are redundant once the tower is in.

## 3. Other findings
- Taste strength vs activity: the tower's within-AUC rises with history length (0.741 for ≤5 clicks, 0.79–0.81 for >5, 0.814 for a full 20).
  Mean |u| falls with length (1.35 down to 1.0), which is just the centroid of fewer items. Activity mainly buys estimation precision.
- Time of day: within-AUC by tb 0/1/2/3 is 0.814/0.802/0.792/0.787 (1,400–1,800 events each, SE about 0.007). This is a weak trend; I did not check it against history length.
- Item bias: there is no transferable signal (see 1b).

## Recommended (ranked)
1. **Two-tower conditional-logit item vectors** (tt.py: `--d 12 --tied 1 --init V_ab55 --nobias 1 --nonull 1 --lr 1e-3 --ep 6`).
   - Recipe:
     - Build the SVD init from all pre-C clicks: binary user × item matrix, scaled by diag(1/sqrt(len_u)) · M · diag(1/sqrt(pop_i+5)), exact svds k=16 random_state 0, sort descending, take comps 2..13 times s, and L2-normalise the rows.
     - Train on train events with ts < C: u = mean of the history items' vectors (last ≤20 clicks); score = u·v; 10-way softmax CE on clicked events; Adam, batch 2048, seeded.
     - Features: dot; and per event, lse, mean and dot − mean of the dot.
   - Expected gain: resid **+0.022/+0.016** (dot alone +0.019/+0.014). Most of this should survive beyond the 325 existing features, since it is measured on top of base.
   - Leak check: vectors and the tower see only train events with ts < C for each cut. For the final submission, fit on all train (< J27) and score eval rows from their own history columns. No eval aggregates. The event aggregates use only that event's own slate.
   - Cost: SVD about 3 s, training about 20 s on 2 CPU threads, RAM < 2 GB.
2. **Fixed SVD with row normalisation** (beta 0.5, all clicks, k13): +0.013/+0.010. This is the cheap fallback; it is also the init for #1.
3. **Event-level slate affinity for an event or click-occurrence model**: lse and mean of u·v plus user CTR gives event AUC 0.653–0.664 vs 0.61.
   Only useful if the pipeline has an event-level component; otherwise #1's event aggregates already capture it.

Nulls: ALS, per-user ridge, free user vectors, item biases, learned null head, d≠12, and untied towers.
Caveat: model and epoch choices were made while looking at val. Curves were flat over epochs 3–7 and across seeds, but re-confirm on full val with a per-cut refit.
