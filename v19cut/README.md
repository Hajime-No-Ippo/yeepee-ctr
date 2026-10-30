# v19cut: feature drop-set screen (dwbig LightGBM, 1000 rounds, lr .015, 4 threads pinned to cores 4-7, shared box)

Baseline seed 0 (from v19 logs, 8 threads, free box): B C25 .81742 C23 .79647; A C25 .81751 C23 .79679.

| var | set | nf (dropped) | C25 s0 | C25 s1 | C23 s0 | C23 s1 | dC25 | dC23 | fit_s C25 (s0/s1) | fit_s C23 |
|---|---|---|---|---|---|---|---|---|---|---|
| A | S | 401 (83) | 0.81737 | 0.81758 | - | - | -0.00004 | - | 694/272 | -/- |
| A | base | 484 (0) | - | 0.81751 | - | - | +0.00000 | - | -/318 | -/- |
| B | S | 401 (83) | 0.81732 | 0.81753 | 0.79654 | 0.79640 | +0.00009 | -0.00003 | 286/278 | 140/143 |
| B | SP | 382 (102) | 0.81727 | 0.81731 | 0.79627 | 0.79638 | -0.00005 | -0.00018 | 264/264 | 136/135 |
| B | SPC | 370 (114) | 0.81702 | 0.81718 | 0.79630 | 0.79640 | -0.00024 | -0.00015 | 250/254 | 131/132 |
| B | base | 484 (0) | 0.81742 | 0.81726 | 0.79647 | 0.79653 | +0.00000 | +0.00000 | 312/1390 | 154/159 |

Deltas = mean over available seeds of the set minus mean over baseline seeds (seed0 from v19 log + seed1 rerun here). Keep rule: both deltas >= -.00003.

## Notes / verdict
- Dropped counts verified from actual nf: 484 -> 401 (S, 83), 382 (SP, 102), 370 (SPC, 114 = SP + 12 more hist_*/co_*). For B the nmA_ names in Data Wrangler's regexes were mapped to nmB_ (B has no nmA_ columns; plain regex matched only 78/97).
- B baseline seed 0 was rerun here and reproduced the v19 log exactly (.81742 / .79647), so runs are deterministic per seed. Seed noise on baseline C25: .81742 vs .81726 (0.00016).
- Fit times are with 4 threads, pinned cores 4-7, shared box (other jobs ran at times). Clean-ish comparison (C25, s1): base 318s (A) / 312s (B s0), S ~275s, SP ~264s, SPC ~252s. B base s1 (1390s) and A_S s0 (694s) were contention-inflated. C23: base ~155s, S ~141s, SP ~136s, SPC ~131s.
- A branch only has S at C25 (no C23, SP).
- Verdict: S is neutral within noise (B: +.00009 C25 / -.00003 C23, A: -.00004 C25); SP and SPC fail (C23 -.00018/-.00015). Gains from cutting are nil; value is only ~11-14% less training time.
