# v7 error analysis (val25 / val26)

Source: the saved v7 full-data val predictions in `v2work/ens/` (blend = mean logit of base s0/s1 at 400 rounds and slow at 500 rounds). No retraining.
Scripts: `research/v7_err_scripts/err.py` and `err2.py`; raw numbers are in `err.json` and `err2.json`.
Eval segment shares come from the eval **features** only (no labels). Decile edges are fixed on val25 and reused for val26 and eval.

Overall: **val25 AUC 0.81164** (565k rows, CTR 6.34%), **val26 AUC 0.82099**. Event click rate is 63.4% on both.

## Segments (val25 | val26)

Each cell is share / CTR / AUC; the last column is the eval row share. AUC is the global row AUC inside the segment.

| Segment | val25 | val26 | eval share |
|---|---|---|---|
| **User type** | | | |
| warm (has train rows) | 96.3% / 6.34% / 0.8141 | 98.0% / 6.34% / 0.8225 | 96.6% |
| history-only (cold, non-empty history) | 2.9% / 6.55% / 0.7533 | 1.4% / 6.56% / 0.7596 | 2.2% |
| empty history | 0.8% / 4.52% / **0.5648** | 0.6% / 4.55% / **0.5380** | **1.3%** |
| **Time since last user activity** (min of last train event, last click) | | | |
| < 1 h | 0.45% / 6.2% / 0.842 | 0.5% / 5.5% / 0.835 | 0.4% |
| 1–24 h | 47.5% / 6.31% / 0.8252 | 59.6% / 6.35% / 0.8302 | 39.6% |
| **≥ 1 d** | 52.0% / 6.36% / **0.7982** | 39.9% / 6.32% / **0.8061** | **60.0%** |
| **Event staleness** (event time − cutoff) | | | |
| < 1 d | 80.1% / 6.34% / 0.8151 | 100% / 6.34% / 0.8210 | 66.6% |
| 1–2 d | 19.9% / 6.33% / **0.7973** | – | 16.6% |
| ≥ 2 d | – (not in any val set) | – | **16.8%** |
| **Day of week** | Sat 80.1% (0.8151), Sun 19.9% (0.7973) | Sun 100% | **Mon 66.6%**, Tue 16.6%, Wed 7.4%, Thu 4.2%, Fri 2.4%, Sat 1.7%, Sun 1.1% |
| **User activity decile** (clicks to date; 0 = least active) | | | |
| d0 | 7.3% / 5.73% / **0.7196** | 4.9% / 5.56% / **0.7206** | **9.5%** |
| d1 | 10.5% / 6.00% / **0.7662** | 7.8% / 5.98% / **0.7797** | **12.2%** |
| d2 | 11.3% / 6.05% / 0.7903 | 9.0% / 5.98% / 0.7923 | 11.8% |
| d3–d5 | 30.7% / ~6.3% / 0.812–0.822 | 29.2% / ~6.2% / 0.809–0.826 | 29.1% |
| d6–d9 | 40.2% / 6.5–6.8% / 0.825–0.840 | 49.0% / 6.4–6.8% / 0.832–0.843 | 37.4% |
| **Item popularity decile** (impressions to date; 0 = tail) | | | |
| d0 (tail) | 9.3% / 5.61% / 0.7682 | 3.9% / 5.36% / 0.7771 | 1.4% |
| d1–d2 | 20.1% / 5.7–6.0% / 0.789–0.797 | 15.3% / 5.7% / 0.781–0.793 | 10.7% |
| d3–d6 | 40.3% / 6.0–6.5% / 0.808–0.821 | 42.1% / 5.9–6.5% / 0.806–0.823 | 41.1% |
| d7–d9 (head) | 30.3% / 6.6–7.1% / 0.819–0.831 | 36.7% / 6.7–6.9% / 0.827–0.843 | **46.8%** |
| **Event outcome** | | | |
| event has a click | 63.4% / 10.0% / 0.8150 | 63.4% / 10.0% / 0.8243 | – |
| event has no click | 36.6% / 0% / – | 36.6% / 0% / – | – |

There is no position-in-event signal to analyse. rowid and event_id are random, and the guideline says they carry no information. There is no slot column, so this segment can't be built.

## Between-event vs within-event decomposition

| | val25 | val26 |
|---|---|---|
| Event-level AUC (has any click; score = 1 − Π(1 − p)) | 0.6689 | 0.6746 |
| Within-event AUC (clicked row's mean rank among its 10) | 0.8003 | 0.8105 |
| Top-1 hit rate in clicked events | 37.9% | 39.9% |
| Global AUC if within-event order were perfect (swap oracle) | 0.9671 (**+0.1555**) | 0.9684 (**+0.1474**) |
| Global AUC if event labels were known (no-click events pushed to bottom) | 0.8874 (+0.0758) | 0.8930 (+0.0721) |

Two-thirds of the remaining error is within-event (which candidate gets clicked); one-third is between-event (whether the event gets a click at all). The event-level AUC of about 0.67 is weak, but between-event gains are capped at about +0.07.

## Eval-mix reweighted val AUC

The val rows are reweighted so each segment's share matches eval.

| Reweighting | val25 | val26 |
|---|---|---|
| none | 0.81164 | 0.82099 |
| last-activity mix (60% ≥ 1 d) | **0.80945 (−0.0022)** | **0.81613 (−0.0049)** |
| user-type mix | 0.81155 | 0.8197 (−0.0013) |
| staleness mix (only < 2 d is representable) | 0.81163 | 0.82099 |

## Top 3 bottlenecks

Weight = eval share × (overall val AUC − segment AUC), averaged over the two cutoffs. The segments overlap, because low-activity users are also the long-gap users.

1. **Low-activity users (activity d0–d2): weighted deficit about 0.017.** AUC is 0.72 / 0.77 / 0.79 against 0.81–0.82 overall, and these deciles are 33.5% of eval vs 22–29% of val. Evidence: d0 AUC 0.720 on both cutoffs, 9.5% of eval.
2. **Long-gap or stale users (last activity ≥ 1 d; event ≥ 1 d after cutoff): weighted deficit about 0.008, plus an unmeasured tail.** Long-gap AUC is 0.798 / 0.806 vs 0.825 / 0.830 for 1–24 h, and this group is 60% of eval vs 40–52% of val. Evidence: reweighting val to eval's gap mix alone costs −0.0022 / −0.0049. Another 16.8% of eval is ≥ 2 d stale, which no val set covers.
3. **Cold and empty-history users: weighted deficit about 0.005 on 3.4% of eval.** Empty-history AUC is 0.54–0.56, close to random; history-only AUC is 0.75–0.76. Evidence: eval has 1.6–2.2× val's empty share (1.26% vs 0.56–0.82%).

**Structural:** within-event ranking holds two-thirds of the error (oracle +0.15 vs +0.07 for event-level). The lever is choice-aware, listwise features and models, not event-level calibration.

**Monday (66.6% of eval) can't be tested.** Every val set is weekend-only (val25 is Sat/Sun, val26 is Sun), and the training folds f1–f6 are Tue–Sun. Eval's Monday rows are exactly its < 1 d stale rows. Monday is a weekday like the Tue–Fri training targets, and earlier work found CTR flat by day and time bucket (auc_ideas_v1), so the risk is probably small. It is the largest unmeasured distribution shift.

Item tail is not a bottleneck for test: eval is head-heavy (46.8% in the top 3 popularity deciles vs 30–37% in val), so the weighted deficit is about 0.003.
