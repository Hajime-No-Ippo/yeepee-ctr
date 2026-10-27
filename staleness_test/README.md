# Staleness test (A: staleness features, B: delay-matched training rows) — 27 Oct 2026
Verdict: NEITHER HELPS. All deltas vs baseline are between -0.0001 and -0.0036 reweighted AUC on both folds (decision rule needed >= +0.002).
Scripts: build.py (features, ~35 s), train.py (LightGBM, 7 variants x 3 seeds x 2 folds), summarize.py (tables + event bootstrap). Raw: results_raw.csv, summary.txt, preds_C24/25.npz.
Data: lite/train_*.parquet (read-only). 100% of users (no subsample). No eval rows used anywhere.

## Results (mean ± sd over 3 seeds; Δ = paired per-seed diff vs base)
Reweighted AUC = eval staleness mix 67/16.5/16.5 (C=25 has no >=48h rows -> 67/33).
| fold | variant | global AUC | rw AUC | 0-24h | 24-48h | 48h+ | Δ rw (seeds) | Δ rw seed-avg [95% bootstrap CI] |
|---|---|---|---|---|---|---|---|---|
| C25 | base | .7750 | .7732 | .7777 | .7636 | – | – | – |
| C25 | +A | .7747 | .7728 | .7776 | .7626 | – | -.0004±.0004 | -.0003 [-.0006,-.0001] |
| C25 | +Atime | .7745 | .7726 | .7775 | .7624 | – | -.0006±.0003 | -.0005 [-.0008,-.0002] |
| C25 | +B | .7750 | .7731 | .7777 | .7635 | – | -.0001±.0002 | -.0001 [-.0004,+.0002] |
| C25 | +A+B | .7747 | .7727 | .7776 | .7624 | – | -.0005±.0005 | -.0004 [-.0007,+.0000] |
| C25 | +B_w | .7744 | .7724 | .7774 | .7620 | – | -.0008±.0002 | -.0007 [-.0011,-.0001] |
| C25 | +A+B_w | .7735 | .7712 | .7769 | .7593 | – | -.0020±.0001 | -.0017 [-.0023,-.0011] |
| C24 | base | .7665 | .7651 | .7723 | .7503 | .7495 | – | – |
| C24 | +A | .7661 | .7646 | .7720 | .7499 | .7480 | -.0005±.0002 | -.0004 [-.0007,-.0001] |
| C24 | +Atime | .7662 | .7648 | .7722 | .7499 | .7485 | -.0003±.0003 | -.0002 [-.0005,+.0000] |
| C24 | +B | .7664 | .7650 | .7722 | .7504 | .7498 | -.0000±.0002 | -.0001 [-.0003,+.0002] |
| C24 | +A+B | .7660 | .7644 | .7723 | .7488 | .7468 | -.0007±.0002 | -.0006 [-.0010,-.0002] |
| C24 | +B_w | .7660 | .7645 | .7721 | .7498 | .7480 | -.0005±.0003 | -.0006 [-.0009,-.0002] |
| C24 | +A+B_w | .7640 | .7615 | .7718 | .7470 | .7360 | -.0036±.0005 | -.0030 [-.0036,-.0024] |
Seed sd of a single model ~0.0001-0.0003; noise on paired diffs ~0.0002-0.0005.
The staleness AUC drop is real (C24: .772 -> .750 -> .750) but neither fix closes any of it.

## Training/validation rows (fold C, pseudo-cutoffs P = C-3d, C-2d, C-1d)
- Snapshot at X = all stats from train events with ts < X only: click log = dedup (user,item,ts) of those events' history entries + their clicked rows; impressions = their rows. (Later events' histories are NOT used, even for pre-X clicks.)
- plain (used by base/+A/+Atime): for each P, each user's first event in [P, P+1d), features from snapshot P. Pooled -> 0% rows >=24h stale. (Same as v2work's folds.)
- B: for each P, each user's first event in [P, C), features from snapshot P. Pooled stale>=24h ~17% (C25 1.27M rows, C24 1.27M rows vs plain 1.05M). B_w: rows weighted so the pooled 0-24/24-48/48+ mix = 67/16.5/16.5.
- validation: each user's first event in [C, Jan27 00:00), features from snapshot C. C25: 43,964 events (20% >=24h, 0% >=48h); C24: 47,928 (27% >=24h, 8% >=48h).
- LightGBM binary, lr .05, 63 leaves, min_data 200, ff .8, bagging .8/1, l2 10, max_bin 127, 400 rounds, seeds 0/1/2.

## Features (t = event ts, X = snapshot cutoff, H = row's own history, emb = SVD item vectors)
Base: hist_len; warm_share (H share with ts < Jan20); inh_cnt (count of candidate in H); inh_age_h (t - last ts of candidate in H, h); inh_pos_end;
i_clk / i_clk24 / i_clk72 (log1p item clicks in log before X: all / last 24h / last 72h); i_imp (log1p impressions); i_ctr (smoothed, prior 20);
u_ev, u_clk (log1p user events / logged clicks), u_ctr (event click rate, prior 5 @ .629);
sim = cos(emb[cand], mean emb over H); sim_rec5 / sim_old (last 5 vs older H items); sim_dec48 (weights exp(-(t-ts)/48h));
emb: binary user x item matrix from log before X, scaled 1/sqrt((n_u+1)(n_i+1)), svds k=16, item vec = V*s, L2-normalised (sim alone = .755 AUC on C25);
co-click: pair counts of item pairs at lag 1-2 in each user's time-ordered click log before X; cc_sum = Σ_h log1p(c), cc_nsum = Σ_h c/sqrt((clk_h+1)(clk_c+1)), cc_nmax;
event-level: rank within event of sim, cc_nsum, i_clk, inh_cnt, sim_dec48; ev_sim_max, ev_sim_mean, sim_gap (sim - event max), ev_inh_n, ev_cc_max.
A: stale_h = (t-X)/3600; A_since_click_h = t - last H ts; A_click_to_cut_h = X - last H ts; A_since_ev_h = t - user's last event before X;
A_hist24 (H clicks in last 24h before t); A_hist_span_h (t - first H ts); A_inh_x_stale (stale_h if candidate in H else NaN);
A_inh_age_rel = inh_age_h / A_since_click_h; A_sim_diff = sim_rec5 - sim_old. +Atime = A without A_sim_diff.
Caveat: recent-vs-old similarity and decayed similarity are already in base (that is where they helped), so A tests the extra staleness/time features.
