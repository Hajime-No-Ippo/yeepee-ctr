# Huawei TechArena 2026 Phase 1 CTR: data profile
Generated 2026-10-23 02:20 IST (Europe/Dublin). Engine: DuckDB 2.0 via `profile.py`. Intermediate data: `work.duckdb`, which holds the parsed train/test tables plus `h_items`/`h_ts` list columns.

## 1. Transfer and files
- `train.zip` = `cat parts/train.zip.part_aa..ag` (7 parts, 650,123,679 B). ZIP64 with EOCD present; `unzip -t` OK. The parts stopped changing after 02:12:30.
- `evaluation.zip` (44.8 MB): `unzip -t` OK.
| file | size | rows (excl. header) | cols |
|---|---|---|---|
| data/train.csv | 7,261,257,246 B | 6,868,240 | 10 |
| data/evaluation.csv | 458,854,688 B | 529,390 | 9 (no `clicked`) |
| example_submission.csv | 25,410,748 B | 529,390 | 2 |
| submission_v1.csv | 25,410,748 B | 529,390 | 2 |
Also present: INSTRUCTIONS.md, starter_solution.py.

**Columns:** `rowid, event_id, user_id, item_id` (opaque strings), `timestamp` (int Unix seconds, whole minutes), `time_bucket` (0-3), `day_type` (0/1), `history_item_ids`, `history_timestamps` (JSON arrays), `clicked`.
- There is a timestamp.
- There is **no slot/position column** (row order inside an event is random; CTR is flat by order).
- There is **no item, user or context metadata**. Only IDs, time and click history.
- No nulls in any column. Empty histories (`[]`): 22,440 train rows (0.33%), 6,650 eval rows (1.26%).

## 2. Key numbers (train)
- Positives 432,023 / 6,868,240, so **CTR = 0.06290156**. This matches the example-submission constant 0.06290156 exactly, so the example is the train base rate.
- 686,824 events. **Every event has exactly 10 candidates** with distinct items. Each event has one user, one timestamp and one history.
- Clicks per event: 0 clicks in 254,801 events (**37.1%**), 1 click in 432,023 (62.9%), **>1 click in 0 events**. The "at most 1" claim holds, so per-event CTR is 0.629 and per-row CTR is 0.0629.
- Users: 53,003, averaging 13.0 events each (max 521). 2,564 users never click.
- Items: 50,000, averaging 137 impressions each (max 1,739). Item volume vs CTR corr = 0.16.
- Time: 2025-01-20 00:00 to 2025-01-26 23:59 UTC. About 980k rows/day, with daily CTR between 0.0626 and 0.0632. CTR is flat across time_bucket and day_type (all ~0.063).
- History length: mean 17.0, median 20, capped at 20 (67.5% of rows are at the cap). 76% of rows repeat an item inside their history.
- CTR rises with history length: 0.043 with no history up to 0.064 at 20.
- **Candidate already in the user's history: CTR 0.293 vs 0.057** (2.6% of rows). This is the strongest single signal found so far.
- Duplicates: 0 full-row, 0 ignoring rowid, 0 (event_id, item_id).

## 3. Train vs evaluation
- Eval schema is identical minus `clicked`.
- **Eval is strictly later in time**: 2025-01-27 00:00 to 2025-02-02 23:56 UTC, with no overlap. It holds each user's *first* event after the cutoff: 52,939 events, one per user. It is front-loaded: 67% of rows fall on Jan 27 and only 2.8% on weekends (train: 28.6%).
- User overlap: 97.3% of eval users appear in train. 1,453 are cold users: 294 of them have empty history and 1,159 have history only. 1,517 train users are absent from eval.
- Item overlap: all 49,892 eval items appear in train.
- Gap from a user's last train event to their eval event: p10 8 h, median 31.5 h, p90 121 h.
- Eval history is shorter (mean 13.4 vs 17.0). Candidate-in-history rate is 2.5%, close to train's 2.6%.
- **example_submission rowids match evaluation.csv exactly**: same 529,390 ids, same order, no duplicates. Header is `rowid,predicted_probability`.
- **submission_v1**: same ids in the same order, all values valid in [0,1], 475k distinct values. It is **exactly uniform on [0,1]**: mean 0.5000, deciles about 52-53k each, and per-event sums average 5.0. So it is a global percentile-rank transform or random scores. Either way it is fine for AUC but uncalibrated. If it is not a deliberate rank transform of a model, check that it isn't random. Chart: `charts/submission_v1_hist.png`.

## 4. Leakage and history semantics
- No column is derived from the label in the same row. History is strictly before the event: no history timestamp ≥ the event timestamp, and no history entry ever equals the current event's click. Histories are sorted oldest to newest. Item and timestamp arrays are always the same length.
- **Warm-up clicks can be told apart** by timestamp: `h_ts < 1737331200` (2025-01-20). Warm-up history starts 2025-01-06. Of the unique-event history entries, 6.50M are warm-up and 5.18M are in the train window.
- **History is an exact click log.** All 431,081 in-window history entries (unique user, item, ts) match a train click row. 431,081 of the 432,023 train clicks appear in some later history (train or eval). Eval histories contain train-period clicks.
- **The main leakage risk is cross-event.** For any event except a user's last, the label is recoverable from the user's *next* event's history: which item was clicked, and whether any click happened. Any feature that joins later events (future history, "next event" stats, full-period user/item CTR computed without a time cutoff) leaks. Eval has no next event, so such features would inflate validation and fail on the leaderboard.
- Aggregate target stats (user/item CTR) must be computed only on events before the validation/eval cutoff.

## 5. Top anomalies / concerns
1. Cross-event label leakage through later histories (see §4). This is the main validation trap.
2. Eval is "first event per user after the cutoff". It is time-skewed toward Jan 27, has more weekdays, and the gap since the user's last event varies. A random or last-day split doesn't reproduce this.
3. There are no side features: no position, no item or user metadata. The signal must come from IDs, collaborative filtering and history (repeat clicks, co-click, recency).
4. History is truncated to the latest 20, so for heavy users it underrepresents their past. Also, 1,453 eval users are cold and 294 of them have no history at all.
5. submission_v1 is perfectly uniform, so check whether it is a real rank transform or a bug/random file. Also: 37% of events have no click, so per-event softmax-style normalisation is not appropriate.

## 6. Suggested next steps
1. **Validation:** use a time cutoff, not a user split, because eval users are 97% train users.
   - Fit on events before 2025-01-25 00:00 (or 01-26).
   - Validate on **each user's first event at or after that cutoff**, which mirrors eval.
   - Keep whole events together and score with global ROC AUC.
   - Compute all label stats on the fit window only.
   - Optionally also report AUC on all post-cutoff events.
2. **Features without labels:**
   - Rebuild the user's click sequence from history plus pre-cutoff train clicks.
   - Candidate-in-history count and recency (time since the last click on this item).
   - Item popularity and recent trend (clicks and impressions in the last N hours before the event).
   - Item-item co-click / sequential similarity of the candidate to recent history items.
   - User activity, history length, time since last click, and warm-up vs in-window share.
   - Rank of the candidate within its event on each feature.
3. **Models:** a LightGBM/XGBoost ranker or binary classifier over those features (group = event). Compare with an MF/two-tower or sequential model (SASRec-like) on history and candidates, then blend by rank.
4. **Final fit:** retrain on the full train window with features recomputed at the cutoff. Check the submission against `example_submission.csv`: 529,390 unique ids, values in [0,1], no NaN.
5. Replace or verify submission_v1 before using one of the 10 daily slots.

Charts: `charts/label_balance.png`, `charts/cardinality_history.png`, `charts/submission_v1_hist.png`.
