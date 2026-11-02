# gfeat -- scoring Grok Bot's v21gfeat (top-80 LGB + 3-seed graph feats) on top of eq4 + graph0312 (Code Buddy, Nov 2 2026, Dublin time)
Authorised by Eric (follow-up "Yes do the both"). v21* read-only; CPU 4 threads; no uploads/submissions/commits. Script scoregf.py, raw numbers results.json/grids.npz/run.log, frozen copies + SHA256.txt of gfeat_c*.npy and gfeat_control_c*.npy. blend_oof.py not modified.

## Alignment
float32 raw scores, lengths 501550/439640/351380 == val tables; rowid val{c}.parquet == tab_val{c} == oof_c{c}.npz, y identical, all finite: OK on C23/25/26 (both gfeat and gfeat_control). Global rank/N only.

## Standalone AUC / rho (rank, every 3rd row)
| fold | gfeat (F) | control (same LGB, no graph feats) | M | graph0312 | rho F~M | rho F~graph0312 | control~M |
|---|---|---|---|---|---|---|---|
| C23 | .79751 | .79673 | .79862 | .76818 | .975 | .851 | .976 |
| C25 | .81765 | .81625 | .81904 | .79514 | .980 | .887 | .980 |
| C26 | .82655 | .82478 | .82797 | .80560 | .980 | .908 | .982 |
(gfeat is +.0008/+.0014/+.0018 over its own control standalone; the graph features are what matter.)

## Held-out test. Score = (1-wg-wf)*base + wg*graph0312 + wf*gfeat (rank/N). base = eq4(z,M,big,D) at C23/C25, M/big 50-50 at C26. Grid wg .01 step, wf .02 step. Weights fit on the source fold, scored on the target fold. Comparator = same base + graph with wg fit on the same source fold.
| target <- fit | eq4+graph (wg) | joint (wg, wf) -> AUC, gain | wf only, wg fixed -> AUC, gain |
|---|---|---|---|
| C25 <- C23 | .82025 (.13) | (.11, .20) .82036, **+1.0e-4** | (.13, .20) .82037, **+1.1e-4** |
| C23 <- C25 | .79978 (.15) | (.13, .16) .79995, **+1.7e-4** | (.15, .16) .79990, **+1.3e-4** |
| C26 <- C23 | .82865 (.13) | (.11, .20) .82889, +2.4e-4 | (.13, .20) .82889, +2.4e-4 |
| C26 <- C25 | .82865 (.15) | (.13, .16) .82887, +2.2e-4 | (.15, .16) .82885, +2.0e-4 |
User-hash (8 groups) delta of the joint held-out candidate vs eq4+graph: C25 +1.04e-4 (sd across groups 0.99e-4), C23 +1.69e-4 (sd 0.83e-4): consistently positive, not noise, but small.
Control LGB (no graph feats) in the same test: C25 -0.4/-0.2e-4, C23 +0.3e-4, C26 -0.0..+0.2e-4 => the gain is from the graph features, not from just another LGB.

## In-sample (oracle on each fold) joint vs wg-only, to check Grok Bot's numbers
C23: .79979 -> .79997 (wg .11, wf .20) = **+1.7e-4**; C25: .82026 -> .82037 (.13, .16) = **+1.1e-4**; C26 (vs M/big+graph) .82865 -> .82890 (.11, .24) = **+2.4e-4**. Grok: +16.6 / +9.8 / +22.8 e-5. **Confirmed** (differences <=0.6e-5 from grid granularity). Held-out gains are as large as in-sample, so little overfit in the weights, but the gain itself is small.

## gfeat replacing part of M and/or big (graph weight refit on source fold, gain vs eq4+graph at its held-out wg; C25<-C23 / C23<-C25)
| variant | gain e-4 |
|---|---|
| F replaces M 50% | +0.9 / +1.1 |
| F replaces big 50% | +0.8 / +1.1 |
| F replaces M 100% | +0.7 / +0.6 |
| F replaces big 100% | +0.4 / +0.6 |
| F replaces M+big 50% each | +0.6 / +0.6 |
| F replaces M+big 100% | -3.7 / -4.4 |
Replacing is no better than adding on top.

## Verdict vs bar (>= +2e-4 held-out on BOTH C23 and C25, not negative on C26)
**FAIL.** Best held-out: C25 +1.0..+1.1e-4, C23 +1.3..+1.7e-4 (about half / 2/3 of the bar); C26 passes (+2.0..+2.4e-4, positive). Direction is consistently positive and group-stable, so it is a real but sub-bar gain (~+1e-4). Optimal gfeat weight is large (wf .16-.20 with wg .11-.13) because it is ~.975 correlated with M but better than a plain LGB.
## Caveats
- gfeat is a single LGB seed, graph features from only 3 GCN seeds (0,1,2) for ALL tables (train f1..f6 from graphs k=1..6, val from k=3/5/6) versus 7 seeds in graph0312; a 7-seed rerun was not done (Grok: ~70 more min of GCN). More seeds might help the feature a bit but the bar is 2x away.
- Rounds (600 C23/C26, 900 C25) were chosen with val monitors (mild selection). D at C23 is a proxy; C26 has no z/D so its gain is on a different (weaker) base and is not comparable to C23/C25.
- No eval-side gfeat array exists (needs k7 3-seed feats + a train on f-tables); nothing built.
