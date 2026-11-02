# v21graph STATUS (started 2026-11-02 01:53 Dublin)
Goal: collaborative-graph model (LightGCN + slate-softmax) on TRAIN-ONLY click graph at each cutoff; must be >=.815 standalone C25 with rho<=.97 vs M/A and add >=+.0003 locally on C23 and C25.
Code: prep.py/prep2.py (tables + graphs, rowid order asserted == v21seq seqfeat order), gcn.py (LightGCN training), gfeat.py (graph features), evalg.py/diag.py (gain tests).
Graph at cutoff X_k = dedup (user,item) clicks (histories+clicked candidates) of train events ts<J20+k*DAY (k=3 for C23, 5 for C25, 6 for C26, 7 for eval). 50k items, 53k users, 0.78M edges at k=5.
## Log
- 02:00 pure LightGCN d32 L2 BPR+slate-softmax, k=5, val25 AUC: .7176 (1000 steps, lr5e-3, still rising). Slate loss matters (+.010 vs BPR only). lr2e-2: .708. d64 L3 lr2e-2 peaks .7336 @250 steps then overfits (.715 @1500).
## 02:55 resumed (executor). Multi-seed averaging of LightGCN feats is the key: single seed C25 .768 -> 3 seeds .792; blend gain (rank, dm sum3 of s_uv,s_h,cos_mean, z-scored) on .4z+.6M: C25 w.05/.1/.2 +36/+53/+31, C23 (2 seeds) +32/+35/-46 (e-5). Cfg: u1 = d64 L3 lr1e-2 l2 3e-3 BPR+slate, 350 steps (k5,k6,k7), u1b 800 steps (k3). Killed ppr.py (slow, 1 thread; superseded by seeds).
Running: sweep5.sh (seeds1-2 all cuts + k7 s0-2), sweep6.sh (seeds 3-8 on k5,k3,k6). ms.py = multi-seed blend test (python ms.py 25 23 26); saves out/graph_simple_c{c}.npy.
## 03:15 interim: seeds C25 n=5, C23 n=4 (dm sum3, rank blend on .4z+.6M, e-5): C25 w.05/.1/.15/.2: +38/+60/+65/+54 ; C23 +37/+52/+44/+14. On .3z+.3M+.4D: C25 +41/+64/+71/+62; C23 +44/+64/+62/+37. Standalone C25 .796 (rho_M .871), C23 .765 (rho_M .81). => already PASSES bar at w=.1 (both cuts >= +3e-4). More seeds running (sweep6/7: k5,k3,k6 seeds 3-14; sweep8: k7 seeds 3-10 for eval).
Caveat: C23 step (800) / C25 step (350) chosen with val monitors (mild selection). Graph at cutoff uses train events only (ts<cutoff).
## 03:10 per parent: use RAW (non-demeaned) graph score, same seed set for folds+eval. SEEDS={0,1,2,3,4,5,9} (7) for k3(u1b,800), k5/k6/k7 (u1,350). Killed sweeps 6/7/8; running k7 s3,4,5,9 (sweep9) + finishing k6 s5. Code Buddy's independent test (cb_blend/u1u2/test_graph.log) confirms gains.
## 03:40 FINAL (7 seeds {0,1,2,3,4,5,9} for ALL cuts + eval, raw non-demeaned)
Score G = z(s_uv)+z(s_h)+z(cos_mean), each averaged over 7 seeds, z = per-table mean/std (linear, no labels). u1 d64 L3 lr1e-2 l2 3e-3 BPR+slate-softmax, 350 steps (k5 C25, k6 C26, k7 eval); u1b 800 steps (k3 C23). Graph = TRAIN-ONLY click graph at cutoff (ts<J20+k*DAY); no eval aggregates/labels. (steps chosen w/ val monitors: mild selection)
Standalone AUC: C23 .76818 (rho_M .826), C25 .79514 (rho_M .867), C26 .80560 (rho_M .884)
Gain (e-5) on .4z+.6M, w=.05/.1/.125/.15/.2: C25 +42/+67/+73/+75/+65 ; C23 +42/+63/+65/+63/+42 ; C26 (vs M alone) +40/+63/+69/+71/+62
Gain on .3z+.3M+.4D: C25 +46/+73/+80/+83/+76 ; C23 +47/+72/+76/+76/+58
Fold preds (float32, v21seq val-table row order): out/graph_c23.npy (501550), graph_c25.npy (439640), graph_c26.npy (351380)
Eval: out/graph_eval.npy = V3 ORDER (feat_v3/eval.parquet rowid), 529390 rows; out/graph_eval_v21seqorder.npy (+graph_eval_rowid.npy) = v21seq seqfeat order (neither v3 nor merged).
VERDICT: PASS the local bar (+3e-4 both cuts) ; reweight.py (--order v3 --gain25 .00073 --w25 .125 --gain23 .00065 --w23 .125 --rho25 .867): BAR PASS but weight on S1 is NEGATIVE (-0.045 shrunk; -0.09 unshrunk) -- binormal model thinks graph is too correlated w/ EN (rho .905) to add positively. Contradicts local positive gain; NOT built.
Scripts: final.py (cut scores+report), fin2.py (C26/eval save).
## 03:25 G1 built per Eric: out/probe_v21graph_G1_S1_m045graph7s.csv.zip = nr(S1 - 0.045*nr(graph7s eval)), merged order ids == S1; sha256 845bbeb4...3fd3; copied to Mac Downloads/huawei and verified. Sens table: sens.py. Predicted gain over S1 +1.87e-4 (90% +0.87..+2.89e-4); graph prior +.005 -> negative.

## G2 (after G1 public .828955)
- fitg.py: 11-component binormal public fit (graph flat prior U(.78,.84), rho from eval, noise .004), 5 obs (Q2,R1,S1,T1,G1), sig .00006 tau .00005. Posterior graph public AUC .8064 (5/95 .8031-.8097). Aside: ours A .8194, aD .8181, v5 .7922, v8r .8146.
- Gain vs S1 (x1e-5): w.03 +4.6, .05 +5.5 (5/95 -6.0..+16.9, P>0 .79), .07 +4.9, .09 +2.9, .12 -2.5, .15 -10.2; optimum w=.05 (graph only).
- Joint 6-param refit moves S1 weights to c.138 g.20 q.34 v8 -.154 v5 -.044 wG .09 (+1.4e-4 in-sample); not built (extrapolative).
- Built out/probe_v21graph_G2_S1_p50graph7s.csv.zip = nr(S1+.05 nr(graph)); sha256 e0b2846ef210028df1d73b625dd25fa5f3e1db8d8409791d536e2627bd58ac57; on Mac Downloads/huawei verified (shasum match).
