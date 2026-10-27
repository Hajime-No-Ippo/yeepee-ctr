# v9 time-features: checkpoint notes (times are Europe/Dublin, UTC+0)
- Harness: copy of v9d/val (cl.py now reads the v8-lite base matrix cache read-only from v9d/val/cache_mx and joins new dirs on rowid).
- tm.py builds feat_tmA (decay/window), feat_tmB (time-of-day crosses), feat_tmB2 (raw hr/tb/dt/stale) for all folds in 25 s on 4 cores.
  It uses train rows with ts < C only, plus each row's own h_ts < ts. It never aggregates eval rows. Check: hist>=C share is 0 in every fold.
- cmp9.py reports z8 AUC overall, AUC by staleness bucket (0-24/24-48/48h+), and a reweighted AUC at eval's mix 66.6/16.6/16.8
  (renormalised over the buckets each cut has). Results go to cmp9_results.jsonl.
- batch1 done (13:05-13:50): tA, tB, noTT, agw8, tB2. All |d| <= 0.0003 except noTT (taste dropped): -0.005 to -0.009, and worst on the 24h+ buckets.
- Post-hoc: a staleness offset gives no gain; the optimal sq blend weight is 0.2 in every bucket.
- batch2 running: trimr (seed-noise reference), tAs (11 strongest tmA features).
- batch2 done (13:53-14:20): trimr (seed noise, about ±0.0001) and tAs, which came out slightly negative (-0.0002; 24h+ down -0.0006).
- Conclusion: no time variant beats v8, so no submission_v9 was written. Full table: ablation_table.txt
