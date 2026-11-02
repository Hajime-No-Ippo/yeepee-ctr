# graph0312 -- scoring Grok Bot's 03:12 graph predictions (Code Buddy, Nov 2 2026, Dublin time)
Authorised by Eric ("yes score it"). CPU only, 4 threads, no eval/submission built, no commits/uploads. v21graph read-only.
Script: score0312.py, raw numbers: results.json. Frozen copies of graph_c23/25/26/27.npy are in this folder. blend_oof.py got member `graph0312` (backup blend_oof.py.bak_0312; oof_c*.npz caches untouched).

## What the files are
- Location: /workspace/huawei/v21graph/out/graph_c{23,25,26,27}.npy (not v21graph/ root). float32, v21seq val-table row order.
- Score G = z(s_uv)+z(s_h)+z(cos_mean), each averaged over **7 seeds {0,1,2,3,4,5,9}** at every cut, LightGCN d64 L3 (u1b 800 steps k3 for C23; u1 350 steps k5/k6 for C25/C26). z = per-table global mean/std.
- **NOT per-slate demeaned (raw).** Check: std of per-slate means is 0.93-0.94 vs global sd 2.86-2.93 (a demeaned score would have 0). So it is rule-compliant: global rank/N only.
- Alignment: lengths 501550 / 439640 / 351380 match; rowid of val{c}.parquet == tab_val{c}.npz == oof_c{c}.npz rowid, and y identical, all True for C23/25/26. Eval (c27): 529390 rows == tab_eval rowid length, rowids == graph_eval_rowid.npy, c27 == graph_eval_v21seqorder.npy, finite. (Not used for anything.)
- Previous graphU1 (my u1u2 build) = DEMEANED per slate, seeds C23 5 / C25 7 / C26 5. I also scored its raw (non-demeaned) variant from the same per-seed caches ("GU1raw").

## Standalone AUC and Spearman rho vs M
| fold | G (0312, raw 7s) | graphU1 (demeaned) | GU1raw | M | rho_M: G / GU1 / GU1raw |
|---|---|---|---|---|---|
| C23 | .76818 | .76844 | .76476 | .79862 | .826 / .823 / .811 |
| C25 | .79514 | .79835 | .79514 | .81904 | .867 / .879 / .867 |
| C26 | .80560 | .80783 | .80421 | .82797 | .883 / .886 / .878 |
G on C25 is identical to GU1raw (rho .9999997, same 7 seeds); on C23/C26 G is better than GU1raw only because it has 7 seeds instead of 5. Demeaning per slate adds ~+.003 standalone but, as below, little in the blend.

## Blend (rank/N): (1-w)*equal(z,M,big,D) + w*graph. Baseline equal-4: C23 .79911, C25 .81952
| graph | .225x4+.10 C23 / C25 | oracle w (C23, C25) | HELD-OUT: w fit C23 -> score C25 | w fit C25 -> score C23 |
|---|---|---|---|---|
| **G 0312 raw 7s** | .79976 / .82019 | .13 / .145 | w=.13: **.82025** | w=.145: **.79978** |
| graphU1 demeaned | .79968 / .82014 | .12 / .15 | w=.12: .82019 | w=.15: .79966 |
| GU1raw (5/7/5 seeds) | .79974 / .82019 | .12 / .145 | w=.12: .82024 | w=.145: .79973 |
Previous best graphU1 oracle: C23 .79970, C25 .82022 (matches). New G: oracle C23 .79979, C25 .82026 => about +0.9e-4 / +0.4e-4 over graphU1. Gain vs equal-4 at held-out weight: C25 +7.4e-4, C23 +6.7e-4.

## C26 clean check (members z/D do not exist at C26): M/big 50-50 base .82798
| graph | w=.10 | w=.125 | w=.15 | w from C23 / from C25 | oracle |
|---|---|---|---|---|---|
| G 0312 | .82859 | .82864 | .82865 | .13 -> .82865 / .145 -> .82865 | .145 -> .82865 |
| graphU1 (dm) | .82863 | .82870 | .82873 | .12 -> .82869 / .15 -> .82873 | .16 -> .82873 |
| GU1raw | .82857 | .82861 | .82862 | .12 -> .82861 / .145 -> .82862 | .135 -> .82862 |
Gain over M/big base: G +6.7e-4 at held-out weight, graphU1 +7.1e-4 (the "C26 M/big+graph .82870" baseline reproduced). On the same M/big base at C23/C25 (w=.10): G .79942/.81991.

## Verdict
- Better than graphU1 in the 4-member blend on C23/C25 (+0.4 to +0.9e-4 at held-out weight), slightly worse on C26 (-0.4e-4, graphU1 .82873 vs G .82865). Overall: **about the same** (all differences <1e-4); the 0312 file is the better rule-compliant (non-demeaned) member: vs the raw graphU1 it is better on all folds, and it is the one with 7 seeds on every cut, same seed set as eval (c27).
- Held-out weight stable: 0.13 (from C23) / 0.145 (from C25), C26 oracle .145; curves are flat between .10 and .16 (within ~1e-5). Recommend w ~ .125-.145 (not .10) if used on the .225x4 recipe: the held-out gain is ~+0.6e-4 over w=.10 on C25, +0.2e-4 C23.
- No rule issues found. Caveats: step counts were picked by Grok Bot with val monitors (mild selection); D at C23 is a proxy; z/D absent at C26.
