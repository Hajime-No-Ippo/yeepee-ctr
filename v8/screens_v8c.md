# v8 follow-up screens (25 Oct 2026). Reference: shipped v8 blend 0.8*(b+s)/2+0.2*sq; c25/c26/c23 = 0.81321/0.82188/0.79056
1 stacking LR on (b,s,sq), cross-cut: fit25->26 +0.00015, fit26->25 -0.00002, ->c23 +0.00015/+0.00019. +segment interactions: -0.00046 on c25. +own-event context: +0.00034/+0.00014. ~1-3 s.
2 plain L2 LR (546 cols: 377 standardized, clipped +-5, plus 169 NaN indicators; C=0.01, lbfgs, same folds/negfrac): standalone 0.80890/0.81960/0.78639; Pearson vs GBDT 0.9625/0.9698/0.9531.
  w0.2 member: +0.00045/+0.00065/+0.00080; stack4 cross-cut +0.00037 (c25), +0.00064 (c26), +0.0008 (c23). Cost ~45-50 s per cut on 4 cores; ~+55 s est. at cut 27.
3 fe folds (+e1..e4 / d1..d5): GBDT blend +0.00020/+0.00030; with sq: +0.00013/+0.00020. Fit 2.3-2.6x; est. +~700 s at cut 27 (E1..E6 = 2.81M extra raw rows), which breaks 750 s.
4 tm_stale with monotone -1: +0.00006/+0.00001 with sq. Eval staleness tail p95 361k s vs c25 154k / c26 80k / c23 255k; CTR flat across staleness quintiles.
Public: v7 0.814396, v8 0.814419. Paired event-bootstrap SD of the local sq gain: 0.00016-0.00021 (0.0003-0.0004 on a 30% subsample), so the shortfall is not just sampling noise.
