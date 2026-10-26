"""Huawei TechArena 2026 Phase 1 - submission_v6: single self-contained end-to-end pipeline (raw CSV -> features -> model -> CSV).
Laptop-budget config (as v4_lite) + low-rank history-centroid features: 2-component GBDT blend.
  base : LightGBM binary, lr 0.05, 63 leaves, min_data 200, ff 0.7, bagging 0.8/1, l2 10; 2 seeds (42, 43), 400 rounds
  slow : lr 0.03, ff 0.55, 95 leaves, l2 30, 1 seed (42), 500 rounds
  blend: mean of the two component logits (base = seed-mean raw score), sigmoid.
Training rows: folds f1..f6 = first event per user in [Ck, Ck+1d), Ck = Jan21..Jan26, all stats frozen at Ck. Eval stats frozen at Jan27.
Features: v3 features (train-only global/item stats + each row's own history), v2 features, similarity extras (extra.py,
extra2.py), time/gap features dropped ('notime'). NO aggregates over evaluation rows (item frequency among eval candidates,
eval co-occurrence, ...): global/item tables come from training data only; each eval row only contributes its own history to
its own user features; within-event (same 10 candidates) z-scores/ranks are used. rowid/event_id are never features.
feat_lr (lrfeat2.py): per cutoff C, binary user x item matrix of each user's last 40 distinct pre-C history clicks (train events only),
columns / sqrt(pop+5), exact svds; item vectors = components 2..K+1 (popularity component dropped) * s, K in 4,6,8,10,12,16;
cosine(candidate, centroid of the row's own history) (+ minus 0.5/1.0 x centroid of the user's own past skipped impressions),
within-event z and gap-to-max. Eval fitted at Jan27 on train only.
extra.py item-side vectors / user clusters use train-only user vectors (leak-hygiene fix, 23 Oct 17:10).
Local validation, same recipe at cutoff Jan25 / Jan26: AUC 0.80185 / 0.81192 (v4_lite 0.77860 / 0.79429).

Usage:  python solution_v6.py --raw DIR_WITH_train.csv_AND_evaluation.csv [--work WORKDIR] [--threads 4]
                                   [--out submission_v6.csv] [--from STAGE] [--to STAGE]
Needs: python 3.11+, polars, numpy, scipy, scikit-learn, lightgbm (tested: polars 1.x, lightgbm 4.7). ~10 GB RAM, ~25 GB disk.
The module sources are embedded below (SRC) and written to WORKDIR at start-up."""
import os, sys, time, json, argparse, subprocess
