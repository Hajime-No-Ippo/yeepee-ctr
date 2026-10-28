# v13 NOTES (started 2026-10-28 00:17 Dublin) -- base v12 method b: C25 .81723, C23 .79566, C26 .82604
## step 0: cleanup (Eric standing yes, pt108u)
Freed ~17 GB (df used 75G -> 59G): removed v8/cache_mx (7.7G, v8's own cache; v9d/v8b cache_mx used by v12 untouched, verified matrix(eval/val25/f3) loads),
 v2work/st_fit.csv (3.9G), v4/feat (1.9G), v3/feat (0.9G), parallel/lag + parallel_v6/lag (0.85G), v10 feat_sq10_{bs*,ep2*,p,q,e,c} variants (~1G; feat_sq10 kept, v11 links it),
 all __pycache__ in v3/v4/v8/v8b/v9/v9d/v10/v11/v12/tools, /tmp leftovers (chrome temp, scoped_dir*, my pngs/py/json), pip cache.
 KEPT: v8/train.csv+evaluation.csv (-> v2work CSVs, v8 pipeline inputs), v2work feat dirs (v8/feat_rc symlinks), v8b cache_mx, v11 msA/B/C + feat_sqC, v12 feat_dw, dw_signals, ens, out/, v5, parts/zips.
## setup
v13 = copy of v12 code (paths patched), symlinks to v11 feat dirs + v12/feat_dw; ens/ symlinks v12/v11 components. Extra feature dirs go in v13/feat_ev etc.
## note (00:19 Dublin, from parent): Eric uploaded v12 -> public LB 0.817935 (v8 was 0.8144, +0.0035); local C25 gain vs v8 was ~+0.004 => C25 is a reliable guide.
## step 1: headroom experiment exp1.py (event-level any-click model on v12 logits, OOF) -- running (first launch interrupted)
