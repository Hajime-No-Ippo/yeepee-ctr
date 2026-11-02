# v21gfeat STATUS (started 2026-11-02 ~03:25 Dublin)
Goal: graph scores/sims (v21graph LightGCN, 7-seed avg) as FEATURES in the v21cat numeric top-80 LightGBM; compare to blending graph member. Train-only, no eval aggregates, cores 4-7, 2 thr, <4GB.
## Design
- v21cat folds: cut C -> train tables f{max(1,C-24)}..f{C-21} (each = 1 day of events after fold cutoff X_k=J20+k*DAY), validate val{C}. 
- LEAKAGE POINT: graph embeddings saved in v21graph exist only for k=3,5,6,7. A train row from f_j must see a graph from BEFORE its own day (graph k=j), else its own click is in the graph. So I train extra train-only LightGCN embeddings for k=1,2,4 (same recipe: u1b d64 L3 800 steps for k<=3, u1 350 steps for k>=4, seeds {0,1,2,3,4,5,9}) -> v21gfeat/out/emb_*.npz via gcn_k.py (copy of gcn.py, writes here). Table->graph: f1..f6->k1..6, val23->k3, val25->k5, val26->k6, eval->k7.
- Features (mkgf.py): g_suv,g_sh,g_cosm,g_cosx (7-seed means), g_G (fixed-constant sum3), g_Gsd (seed std), g_udeg,g_ideg (train-graph degrees). Scale stable across k (drift.py) -> raw. + slate-relative (demean/max-diff/rank) added in train script.
## Log
- 03:26 started sweep_train_emb.sh (k=4,2,1 x 7 seeds, ~90-200s each, seed-major).
- 03:44 GCN runs are slow under CPU contention (~17 min/seed for k4+k2+k1). 7 seeds would take ~2h. PLAN: stage 1 = 3 seeds {0,1,2} for ALL tables (train+val, self-consistent; graph-member comparator also at 3 seeds); if promising, extend sweep to {3,4,5,9} and rerun with 7 seeds.
- 04:15 (resumed after interrupt) stage1.sh waits for k1 s2, kills sweep, builds 3-seed graph feats (GTAG=_s3) -> cache/gf_s3_*.npy
- 04:22 3-seed graph feats built (cache/gf_s3_*.npy; f1-f6,val23/25/26; sweep killed after seed 2). run_s3.sh trains numg (top80+graph) and num (control) for C25,C23,C26 sequentially -> ens/gf_{var}_s3_c{cut}_s0.npy (cols = 300/600/900 rounds). Evaluator: evg.py (base=eq4 z/M/big/D rank-mean + 7-seed graph member, per Code Buddy REPORT_nov2_cat_graph.md: eq4+graph.1 = C23 .79960 / C25 .82012).
## RESULT stage 1 (04:45) — 3-seed graph feats for ALL tables (train f1..f6 from graph k=1..6, val from k=3/5/6), single LGB seed, 900/600 rounds, top-80 + 8 graph feats + slate-rel
Standalone: C23 .79751 (ctl .79673) | C25 .81765 (ctl .81625) | C26 .82655 (ctl .82478). rho_M .975/.980/.980, rho_G .81/.87/.90.
Gain over eq4 (z,M,big,D) at w.1/.15/.2/.3: C23 +21/+28/+34/+38 ; C25 +18/+24/+28/+29 (control: +11..14 / +2..-16). C26 (vs M/big) +28/+38/+46/+55.
3-way base+graph(+gfeat) beyond base+graph at its best w (7-seed graph member): C23 +16.6e-5, C25 +9.8e-5, C26 +22.8e-5 (grid-best, optimistic). Vs 3-seed graph member (same seeds): +16.5/+10.7/+22.8. Control LGB adds +5/+1/+2.
VERDICT: FAIL the +2e-4 bar on both C23 and C25 (about half of it on C25). No eval array built, no probes. Fold preds (float32, v21seq val order): out/gfeat_c23.npy (600r), gfeat_c25.npy (900r), gfeat_c26.npy (600r); controls out/gfeat_control_c*.npy. 7-seed rerun NOT done (would need ~70 more min GCN for k1,2,4 seeds 3,4,5,9; sweep killed after seed 2; emb in out/).
