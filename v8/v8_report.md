# v8 result (25 Oct 2026) - NOT SHIPPED (bar not met)
Bar: v7 full + 0.002 on both cutoffs = 0.81364 (c25) / 0.82299 (c26). All numbers are lite blends (base + slow), v8 workspace.
| step | c25 | c26 | val23 | decision |
|---|---|---|---|---|
| ctl (v7 lite) | 0.81117 | 0.82077 | 0.78852 | reference |
| S1 +feat_pp (PPMI) | 0.81173 (+0.00056) | 0.82115 (+0.00038) | 0.78895 (+0.00043) | best; kept as candidate, below bar |
| S2 +feat_la (low-activity) | 0.81162 | 0.82109 | 0.78885 | dropped (below pp) |
| S3 GBDT-leaf LR on pp, w0.1 | 0.81153 (LR alone 0.78236) | - | - | dropped |
| lambdarank member on pp, w0.1 | 0.81167 (alone 0.80431) | - | - | dropped |
| within-event z blend tweak | 0.81121 vs 0.81117 | - | - | dropped |
pp corr check vs feat_tt/lr2: max |r| 0.766 -> all 9 kept.
Timing: 2 timed_run_v8 runs (lite_nox2, 4 cores) were contaminated by VM memory ballooning (MemAvailable fell to ~3 GB):
feat_v2 557/507 s vs 119 s clean, slow 198/400 s vs 85 s clean. Unaffected stages: feat_pp 35.5/38.0 s, base 75.0/78.8 s.
Component estimate (clean recheck + pp): about 716 s, under 750 s, needs a clean confirm.

## Sequence model (seq.py) — 25 Oct, NOT shipped (bar 0.81364/0.82299 not met)
pp ctl 0.81173/0.82115. Best single (blend raw w0.2): all(700k,3ep) 0.81325/0.82187; slate 0.81307/0.82191; combo 0.81316/0.82196.
Stacked sq (v2) 0.81304/0.82188. 5-variant logit ensemble raw w0.2: 0.81358/0.82229 (~5 fits, over budget). c26 gain caps ~+0.0011.

## v8 final build (25 Oct 2026, 14:21 Europe/Dublin)
solution_v8 = v7 lite + ppmi (feat_pp) + seq (DIN-style, 'all' variant capped: last 500k train events < C, 3 ep, bs 256, <=6000 steps) blended raw-logit w0.2.
Seq choice: all500 vs combo tied on c25+c26 (1.63508 vs 1.63512); all500 fit 79-81 s vs combo 107-131 s (+~50 s) -> all500.
267 s val26 'all' fit = contention: quiet refit at C26 79 s (589k->500k events capped, 5862 steps); eval fit 81 s.
Speed trims (exact up to float order / existing tie non-determinism): feats.py np.add.at -> sparse matmul, co-occurrence self-join -> sparse B^T B
(peak RSS 6.8 -> 4.5 GB); v3feat_ne ufunc.at -> sparse matmul + sort/reduceat; lrfeat2 / ttfeat np.add.at -> sparse matmul.
AUC (trimmed v7+pp, no seq) c25/c26/c23: 0.81167 / 0.82123 / 0.78885 (pp control 0.81173 / 0.82115 / 0.78895).
v8 final c25/c26/c23: 0.81321 / 0.82188 / 0.79056 (v7 lite 0.81117 / 0.82077 / 0.78852).
Timed (timed_run_v8b, 4 pinned cores, start MemAvailable 14.1 GB, min 9.2 GB, no v5): TOTAL 626.1 s.
