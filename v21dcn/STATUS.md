# v21dcn STATUS (started 2026-11-01 21:46 Dublin)
Goal: raise regularised DCN-v2 standalone AUC + diversity (C23/C25 val, A train-only). Code: dcn21.py (copy of v21fit/div21.py + cr/stack/fmk/wd/lw/candid/xdrop/noise/needseq options, multi-config runner --cfg, data loaded once), run.sh, rep.py (reports), cfg/*.json. ens/{cfg}_A_c{cut}_s{seed}.npy = (rows, epochs) log-probs. rep.py base = .4z+.6M(ctl6 8s): gain = .3z+(.7-w)M+wD.
Run layout: C25 on cores 0-2, C23 on cores 3-5 (3 threads each, ~5GB RSS each). Config list r1 (cfg/r1.json) run in parallel on both cuts.
## Log
- 21:55 r1 started (r0_base reproduction, cid, a1, a01, cl5, cr64, h1024, fm16, stack, wd2, bs512, noslate, xdrop).
