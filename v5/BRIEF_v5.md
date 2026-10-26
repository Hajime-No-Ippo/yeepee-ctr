# Yeepee v5 plan (draft, awaiting Eric's go)

Base: copy of Grok Bot's v2work "ne" pipeline (public AUC 0.7888, val25 0.7775). Work happens in /workspace/huawei/v5 only. v2work and v3 are read-only.
Source of ideas: /workspace/huawei/research/v5_ideas.md (AUC Researcher).

Rules:
- No item or global stats from eval rows.
- Fit before the cutoff; validate on each user's first event at or after the cutoff.
- One heavy job at a time, 3 threads or fewer. Queue behind Grok Bot's ens.py.
- No uploads. Write no submission file unless the decision rule below passes.

## Step A: one extra.py rebuild
1. Contrast-embedding sweep, screened one feature at a time on val25:
   - k in {16, 24, 32, 48}
   - weighting: tf-idf, no-idf, BM25, log-count
   - singular-value scaling
   - negative weight
   Keep the 2-4 best that aren't near-copies (correlation below 0.9).
2. Item-side contrast at k32, plus other negative weights. Fix extra.py L51/L55-58 so Un is built from train-only clicks.
3. Taste-confidence magnitudes:
   - contrast norm before normalisation
   - raw dot
   - click and skip counts
   - raw value minus the event max

## Step B
4. Slate-context features. Pre-check first; drop if the within-event AUC is 0.55 or lower.

## Step C (only if A or B gained)
5. Item2vec-style PMI-SVD from train-only co-clicks. Check for overlap with Grok Bot's rcov/seq.py first.
6. Drop the duplicate feature x_con05_k64. Null-importance pruning, kept only if both cutoffs improve.
7. DART blend member. Lowest priority; coordinate with Grok Bot (ens.py).

## Decision rule
Write /workspace/huawei/v5/submission_v5.csv only if v5 beats the v2work ne baseline (same rows, same seeds) on val25 AND on val26 AND on a weekday cutoff (val22 or val23), by at least +0.001 each. val25 and val26 fall on a weekend, and the eval period starts Monday 27 Jan.

## Report
For each step, a table on val22/25/26 with these columns:
- AUC
- AUC by staleness bucket
- event AUC
- top-15 features
- runtime
