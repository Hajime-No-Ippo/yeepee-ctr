# v21roll — rolling (walk-forward) date validation   [2026-10-31, Europe/Dublin]
No upload, no eval rows/labels used. Existing folders untouched. venv_v17, taskset -c 4-7, 4 threads. All features are the cached point-in-time tables (cutoff = start of the fold's day).

## Files
- `rollval.py`  subcommands: `days` | `plan` | `fit --variant {v12,coA,coB}` | `slate --variant V` (v21slate layer z-.5*lse(z)+.3*P) | `score --names a,b,c --ref a [--pred-dir D] [--boot N] [--tag T]`
  Custom pipeline hook: write `{name}_M{d}_{table}.npy` (scores in cached table row order; `plan` lists the 4 folds / tables) into a dir and run `score --pred-dir D`.
- `report.py` delta table vs public + vs C23/C25 from `score_ref_*.json`.  `logs/` (days.txt, fit_log.jsonl, report.txt), `preds/`, `score_*.json`.

## Day table (train)
shape: (7, 8)
┌─────┬─────┬────────┬────────┬───────┬────────┬──────────┬────────────────┐
│ jan ┆ day ┆ slates ┆ rows   ┆ users ┆ clicks ┆ ctr      ┆ slates_w_click │
│ --- ┆ --- ┆ ---    ┆ ---    ┆ ---   ┆ ---    ┆ ---      ┆ ---            │
│ i64 ┆ i64 ┆ u32    ┆ u32    ┆ u32   ┆ i64    ┆ f64      ┆ f64            │
╞═════╪═════╪════════╪════════╪═══════╪════════╪══════════╪════════════════╡
│ 20  ┆ 0   ┆ 98514  ┆ 985140 ┆ 35259 ┆ 62096  ┆ 0.063033 ┆ 0.630327       │
│ 21  ┆ 1   ┆ 98503  ┆ 985030 ┆ 35156 ┆ 61957  ┆ 0.062899 ┆ 0.628986       │
│ 22  ┆ 2   ┆ 98097  ┆ 980970 ┆ 35112 ┆ 61985  ┆ 0.063187 ┆ 0.631875       │
│ 23  ┆ 3   ┆ 97717  ┆ 977170 ┆ 34912 ┆ 61181  ┆ 0.06261  ┆ 0.626104       │
│ 24  ┆ 4   ┆ 97589  ┆ 975890 ┆ 35040 ┆ 61280  ┆ 0.062794 ┆ 0.62794        │
│ 25  ┆ 5   ┆ 98453  ┆ 984530 ┆ 35230 ┆ 61814  ┆ 0.062785 ┆ 0.627853       │
│ 26  ┆ 6   ┆ 97951  ┆ 979510 ┆ 35138 ┆ 61710  ┆ 0.063001 ┆ 0.630009       │
└─────┴─────┴────────┴────────┴───────┴────────┴──────────┴────────────────┘

## Folds (all models = fast LGB proxy: 300 rounds, lr .06, 63 leaves, ff .5, neg-subsample .35, 435-460 feats, seed 42)
- Training tables f_k = first event/user of day Jan 20+k (35k slates, 351k rows, features cutoff = that day's start, staleness 0-24h).
- Fold d (d=3..6 = Jan 23..26): M_d trained on f1..f{d-1}, tested on f_d (expanding window; test labels are strictly after every training feature cutoff and label; the table's cutoff already enforces the label lag, so no extra gap). K=4 plain folds (3..6); K=3 (4..6) for SM.
- SM_d (d=4,5,6): each day-d event scored with the STALEST cached snapshot that contains it (val23 / val25) else fresh f_d; reweighted to eval mix .6662 fresh / .1655 24-48h / .1682 48h+ (missing bin's mass goes to the present stale bin). Stale events: d=4 24-48h (val23) only; d=5 48-72h only; d=6 both bins (val25 -> 24-48h 6.5k events, val23 -> 72-96h 2.2k events). Few stale events -> noisier.
- C23 = M3 on val23 (50,155 slates), C25 = M5 on val25 (43,964 slates): the existing cuts, same proxy models.
- Weights: mean, lin (1..K), exp (2^(k-1)), normalised. Bootstrap: Poisson(1) per slate, shared across variants (paired), 300 draws.
- Note M6 is expanding (f1..f5), old C26 was sliding f2..f5.

## Per-fold AUCs (proxy; public in last row)
abs AUCs
plain3           0.80299   0.80400   0.80410   0.80443
plain4           0.80943   0.81082   0.81067   0.81085
plain5           0.81876   0.81973   0.81913   0.81965
plain6           0.82437   0.82532   0.82503   0.82523
sm4              0.80482   0.80651   0.80640   0.80658
sm5              0.80708   0.80816   0.80776   0.80833
sm6              0.81728   0.81847   0.81825   0.81847
C23              0.79366   0.79483   0.79486   0.79519
C25              0.81528   0.81628   0.81582   0.81634
plain_K4_mean    0.81389   0.81497   0.81473   0.81504
plain_K4_lin     0.81756   0.81861   0.81829   0.81860
plain_K4_exp     0.81946   0.82047   0.82015   0.82044
plain_K3_mean    0.81752   0.81862   0.81828   0.81858
plain_K3_lin     0.82001   0.82104   0.82067   0.82097
plain_K3_exp     0.82063   0.82165   0.82129   0.82158
sm_K3_mean       0.80973   0.81104   0.81080   0.81113
sm_K3_lin        0.81180   0.81304   0.81278   0.81311
sm_K3_exp        0.81259   0.81381   0.81356   0.81387

n_slates {'plain3': 34912, 'plain4': 35040, 'plain5': 35230, 'plain6': 35138, 'sm4': 35040, 'sm5': 35230, 'sm6': 35138, 'C23': 50155, 'C25': 43964}
mix {'plain3': [1.0, 0, 0], 'plain4': [1.0, 0, 0], 'plain5': [1.0, 0, 0], 'plain6': [1.0, 0, 0], 'sm4': [0.6662, 0.3338, 0.0], 'sm5': [0.6662, 0.0, 0.3338], 'sm6': [0.6662, 0.1655, 0.1683], 'C23': None, 'C25': None}
auc_sd(bootstrap over slates) {'v12': 0.00146, 'coA': 0.00144, 'coB': 0.00143, 'coBs': 0.00142}

## Delta vs public (local delta of the same pair; bootstrap sd of the delta in last column = mean over pairs coA, coB, coBs vs their ref)
pair, public delta: {'v12->coA': 6.4e-05, 'coA->coB': 3.2e-05, 'v12->coB': 9.6e-05, 'coB->coBs': 0.000435, 'coA->coBs': 0.000467, 'v12->coBs': 0.000531}
metric               v12>coA       coA>coB       v12>coB      coB>coBs      coA>coBs      v12>coBs   corr_w_public  ratio(sum loc/sum pub)  sd(boot)
plain3              +0.00101      +0.00010      +0.00111      +0.00033      +0.00043      +0.00144   +0.15    2.72   0.00028
plain4              +0.00139      -0.00015      +0.00124      +0.00018      +0.00003      +0.00142   -0.07    2.53   0.00024
plain5              +0.00097      -0.00061      +0.00037      +0.00052      -0.00009      +0.00089   +0.26    1.27   0.00021
plain6              +0.00094      -0.00029      +0.00066      +0.00020      -0.00009      +0.00086   -0.01    1.40   0.00018
sm4                 +0.00169      -0.00011      +0.00158      +0.00018      +0.00008      +0.00176   -0.09    3.19   0.00024
sm5                 +0.00108      -0.00040      +0.00068      +0.00057      +0.00017      +0.00125   +0.30    2.06   0.00029
sm6                 +0.00118      -0.00021      +0.00097      +0.00021      +0.00000      +0.00118   -0.03    2.05   0.00020
C23                 +0.00118      +0.00002      +0.00120      +0.00033      +0.00036      +0.00153   +0.09    2.84   0.00023
C25                 +0.00100      -0.00046      +0.00054      +0.00053      +0.00007      +0.00107   +0.28    1.68   0.00020
plain_K4_mean       +0.00108      -0.00024      +0.00084      +0.00031      +0.00007      +0.00115   +0.08    1.98   0.00011
plain_K4_lin        +0.00105      -0.00032      +0.00073      +0.00031      -0.00001      +0.00104   +0.07    1.72   0.00011
plain_K4_exp        +0.00102      -0.00033      +0.00069      +0.00029      -0.00004      +0.00098   +0.07    1.60   0.00012
plain_K3_mean       +0.00110      -0.00035      +0.00075      +0.00030      -0.00005      +0.00105   +0.05    1.73   0.00012
plain_K3_lin        +0.00103      -0.00037      +0.00066      +0.00030      -0.00007      +0.00096   +0.07    1.54   0.00012
plain_K3_exp        +0.00102      -0.00036      +0.00066      +0.00029      -0.00007      +0.00095   +0.06    1.52   0.00013
sm_K3_mean          +0.00132      -0.00024      +0.00107      +0.00032      +0.00008      +0.00140   +0.04    2.43   0.00014
sm_K3_lin           +0.00123      -0.00026      +0.00097      +0.00033      +0.00007      +0.00130   +0.06    2.24   0.00014
sm_K3_exp           +0.00122      -0.00025      +0.00097      +0.00031      +0.00006      +0.00128   +0.05    2.22   0.00014

Public probes (p33 w80): v12 .818206, v17cooc A .818270, B .818302, v21slate B .818737. coBs = coB + slate layer (proxy of v21slate; P uses only my row-feature aggregates, no NA/NB features).

## Runtime
fit: 27-54 s per fold (4 folds ~ 2.7 min per variant, 435-460 feats, 275k-686k train rows); slate layer 4 folds ~ 165 s (slate aggregates cached); score with 300 bootstraps ~ 40 s per ref. Whole study (3 variants + slate + 3 scores) < 15 min wall on 4 cores; peak RSS ~ 2-3 GB.

## Findings / recommendation
See final report (kept in the chat); short version below.
1. The public pairs available are tiny (+.00003..+.0001 for cooc, +.000435 slate) vs local proxy deltas ~10x larger for A (+.0010 everywhere): local LGB deltas do NOT transfer to the p33 rank-blend probe magnitude for ANY cut (C23, C25, rolling alike). Local A->B is negative on folds 4-6 and C25 (-.0003..-.0006) while public is +.00003: also inside noise (sd .0002-.0003 per fold, .00011-.00014 for K-fold aggregates).
2. Only pair that exceeds noise and has a sizeable public delta: slate layer (+.000435 public): local +.00033/.00018/.00052/.00020 on folds 3-6 (mean +.00031 K4), C23 +.00033, C25 +.00053: sign and rough size agree for every cut; fold 6 (+.0002) is no closer to public than C25 (+.00053) -> no evidence that later folds track public better.
3. Weighting (mean/lin/exp) changes deltas by <.0001 and cannot be distinguished with the public pairs we have.
4. Rolling K-fold aggregate has lower noise (sd of delta .00011-.00014 vs .00020-.00026 for C25, .00023-.00036 for C23) because it pools 3-4 days. SM costs noise (sd .00014-.00018 for K3) and adds nothing measurable vs plain on these pairs.
