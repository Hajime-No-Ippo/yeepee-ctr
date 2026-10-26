# Jan 25/26 (Sat/Sun) vs weekdays vs eval — Huawei TechArena Phase 1
Generated 2026-10-23 ~23:30 IST (Europe/Dublin). Scripts: build.py, a1–a5.py, charts.py (own DuckDB `j.duckdb`; work.duckdb opened read-only). All dates UTC. No eval aggregates used as features; eval numbers below are descriptive only.

## Bottom line
**Saturday 25 and Sunday 26 are not behaviourally different from weekdays.** Volume, click rate, CTR, hour mix, item popularity and signal AUC all sit inside the weekday range. What *does* differ is (a) where those days sit in the window (later → longer, fresher histories, almost no new users) and (b) how the C=25/C=26 validation sets are truncated (max 48h / 24h since cutoff, vs eval up to 7 days). The validation bias is a **staleness/horizon** problem, not a weekend problem.

## Per day (all train events)
| | Mon20 | Tue21 | Wed22 | Thu23 | Fri24 | **Sat25** | **Sun26** | EVAL |
|---|---|---|---|---|---|---|---|---|
| events | 98,514 | 98,503 | 98,097 | 97,717 | 97,589 | 98,453 | 97,951 | 52,939 |
| active users | 35,259 | 35,156 | 35,112 | 34,912 | 35,040 | 35,230 | 35,138 | 52,939 |
| events/user | 2.79 | 2.80 | 2.79 | 2.80 | 2.79 | 2.79 | 2.79 | 1 |
| event click rate | .630 | .629 | .632 | .626 | .628 | .628 | .630 | – |
| row CTR | .0630 | .0629 | .0632 | .0626 | .0628 | .0628 | .0630 | – |
| CTR in-history / not | .294/.057 | .294/.057 | .299/.057 | .292/.057 | .287/.057 | .294/.057 | .291/.057 | – |
| in-history cand. rate | 2.59% | 2.59% | 2.60% | 2.58% | 2.57% | 2.56% | 2.62% | 2.54% |
| time_bucket 0/1/2/3 % | 10/20/30/40 | 10/20/30/40 | 10/20/30/40 | 10/20/30/40 | 10/20/30/40 | 10/20/30/40 | 10/20/30/40 | 20/28/28/24 |
| mean history len | 16.3 | 16.6 | 16.8 | 17.1 | 17.3 | 17.4 | 17.6 | 13.4 |
| warm-up share of history | .91 | .74 | .62 | .52 | .44 | .38 | .32 | .49 |
| empty history | 0.52% | 0.42% | 0.36% | 0.28% | 0.27% | 0.22% | 0.21% | 1.26% |
| users first seen that day | 100% | 25% | 11% | 6.1% | 4.0% | 2.5% | 1.7% | 2.7% not in train |
| users active only on weekend | – | – | – | – | – | 2.5% | 2.5% | 5.1% of eval users |
| item impr. JS vs other weekdays | .0116 | .0117 | .0118 | .0117 | .0119 | .0113 | .0114 | .0198 vs Mon–Fri |
| item impr. Pearson vs weekdays | .946 | .945 | .945 | .945 | .945 | .948 | .947 | .915 |
| top-100 item impression share | 1.47% | 1.49% | 1.49% | 1.48% | 1.50% | 1.49% | 1.49% | 1.53% |

Item notes: 11.4% of users are weekday-only, nobody is weekend-only except 2.5% of weekend users. JS noise floor: a weekday down-sampled to eval size gives JS 0.0203 vs other weekdays, so eval's 0.0198 is pure sampling noise. Day-level item CTR correlation between days is ~0 (noise), click distributions JS ~0.21 on every day. Hourly event click rate is flat (0.616–0.639) on both weekday and weekend.

## Validation sets (first event per user at/after cutoff) vs eval
| | C=23 | C=24 | **C=25** | **C=26** | EVAL |
|---|---|---|---|---|---|
| events (rows) | 50,155 | 47,928 | 43,964 (440k) | 35,138 (351k) | 52,939 (529k) |
| horizon after cutoff | 4 d | 3 d | 2 d | 1 d | 7 d |
| staleness median / mean / p90 (h) | 16.0/23.5/58 | 15.2/20.4/46 | 13.8/16.6/38 | 11.4/11.5/21 | 16.9/28.7/70 |
| stale ≥24h / ≥48h | 30%/12% | 27%/8% | 20%/0% | 0%/0% | 33%/17% |
| weekend share | 12% | 27% | 100% | 100% | 2.8% |
| time_bucket 0/1/2/3 % | 20/28/28/24 | 21/28/28/23 | 22/29/27/23 | 23/29/27/21 | 20/28/28/24 |
| mean hist len / hist=20 | 12.5/33% | 13.2/37% | 14.0/42% | 15.3/51% | 13.4/39% |
| warm-up share of history | .72 | .65 | .58 | .50 | .49 |
| empty history | 1.5% | 1.2% | 0.8% | 0.6% | 1.3% |
| new users (no train event before cutoff) | 10.0% | 6.0% | 3.3% | 1.7% | 2.7% |
| in-history candidate rate | 2.49% | 2.50% | 2.52% | 2.57% | 2.54% |
| median h since user's last event / last click | 26/40 | 26/37 | 24/33 | 20/25 | 32/43 |
| event click rate | .629 | .629 | .634 | .634 | – |

**Which cutoff looks most like eval?** C=25 is closer than C=26 on almost everything (size, staleness, hour mix, history length, new-user share 3.3% vs 2.7%). C=26 is the least eval-like: only 66% of eval's size, zero rows older than 24h, histories fresher and longer than eval's. C=24 matches eval's staleness and history length best but has twice eval's new-user share (6%) and a staler, warm-up-heavier history.

## Cheap signal AUC (fit on clicks strictly before each cutoff/day; 25% event sample per day)
Signals: in-history flag, log item clicks in the 24h before cutoff, SVD-16 history-similarity cosine (proxy for the taste model).
| set | in-hist | pop | sim proxy |
|---|---|---|---|
| day 21 / 22 / 23 / 24 (weekdays) | .551/.552/.550/.549 | .526/.527/.527/.526 | .740/.753/.762/.768 |
| **day 25 / 26 (weekend)** | .548/.552 | .524/.524 | **.768/.770** |
| val C=24 all / <24h / ≥24h | .550 | .525/.526/.520 | .748/.754/.732 |
| val C=25 all / <24h / ≥24h | .551 | .525/.526/.519 | .755/.758/.743 |
| val C=26 (all <24h) | .551 | .520 | .763 |
- Weekend days score the same as Friday (±0.002, inside sampling noise). The rise from day 21 → 24 is the fit window growing, not day-of-week.
- Staleness costs ~0.015–0.022 sim AUC from <24h to 24–48h — and it happens **inside the all-weekend C=25 set**, so it is staleness, not weekend. Popularity also decays (.526 → .519).
- C=24's "weekend" rows are exactly its ≥24h rows, so weekday-vs-weekend splits at C=24 are confounded with staleness.

## Implications for validation
1. Weekend cutoffs do **not** bias AUC through weekend behaviour; no need for weekday-only or weekday-weighted validation (and restricting to weekday events at C≤24 would *drop* stale rows and make it less eval-like).
2. They **do** bias it through horizon truncation: C=26 has no rows >24h old (eval 33%), C=25 none >48h (eval 17%). Expect C=26 AUC to read ~0.005–0.01 high vs eval, and C=25 a few thousandths high, other things equal (offset partly by eval having one more day of fit data).
3. Fixes: (a) report AUC by staleness bucket (0–24h, 24–48h, 48h+) and reweight to eval's 67/17/17 mix; (b) prefer C=25 over C=26 as primary, and add C=24 (or C=23) as a fold that actually contains 48h+ rows; (c) check feature/model gains on the ≥24h slice, since a feature that only helps fresh rows will over-promise on C=26; (d) new-user share is small everywhere (1.7–6%), so it is a minor factor.

## Charts
- `01_per_day.png` — per-day volume, click rate, in-history CTR, history length/warm-up share, new-user share, item JS.
- `02_hour_mix.png` — hour mix of all events per day vs first-event sets vs eval.
- `03_val_vs_eval.png` — staleness CDFs (C=23–26 vs eval) and key share comparison.
- `04_signal_auc.png` — signal AUC by day and by validation staleness slice.
Data: per_day.csv, hour_mix.csv, items_per_day.csv, valsets.csv, signal_auc.csv.
