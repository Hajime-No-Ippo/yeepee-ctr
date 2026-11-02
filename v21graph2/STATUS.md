# v21graph2 STATUS (reconstructed 2026-11-02 04:15 Dublin; job resumed by executor)
Goal: stronger graph member than the 7-seed sum3 (v21graph). BAR: >= +2e-4 blend gain over the 7-seed baseline on BOTH C23 and C25, at best w, over .4z+.6M and .3z+.3M+.4D.
If pass -> fitg.py (7 public scores) weight on top of G2 -> ONE probe zip only if pred gain vs G2 >= +5e-5 and P>0 > .75 -> Mac ~/Downloads/huawei (shasum). No uploads by me.
Public: EN .828856 Q2 .829016 R1 .829058 S1 .829101 T1 .829096 G1 .828955 G2 .829113 (best).
Rules: train-only graph, cores 4-7 (shared w/ v21gfeat: leave gcn_k.py runs alone), threads 2-4, <5GB, don't touch v21graph.

## Code (reconstructed)
gcn.py = LightGCN+slate softmax (v21graph copy); gcn2.py = + --tau (time-decayed edge sampling for BPR); ii.py = item-item co-click (sessions W sec) feats on hist; iitest*.py = ii feature AUC/pop tests; ana.py = avg(tag,c,seeds,step)/sum3/line() gain table (rank blend w in .05..2 on bases .4z+.6M, .3z+.3M+.4D); combo.py = score-combo search (s_uv,s_h,cos_mean,cos_max weights); cmp1/2 = variant compare; final.py/fitg.py/mkG2.py = fit & probe build (G2 built earlier).
Configs: u1 d64 L3 350st (k5=C25,k6=C26,k7=eval), u1b 800st (k3=C23); u3 d128 L3 (500st/u3b 1100); u4 d64 L2; u5 d128 L2; u6 d64 L4; u7 d64 L3 tau5 (time-decay edges).
Cutoff graph k: C23=k3, C25=k5, C26=k6, eval=k7.

## State at 04:15
- Seeds on disk: u1 k5 s0-11 (12), u1b k3 s0-11 (12), u1 k6 s0-5,9 ; k7 s0-5,9 (7 seeds) -> eval/C26 only 7 seeds so far.
- Variants run on k5 only so far: u3 s0 (d128 L3), u4 s0-2 (d64 L2). u4b/u3b/u5/u5b k3 runs were killed (no emb).
- RUNNING: runq2.sh jobs3.txt (gcn2.py u6 L4 d64 s0,s1 on k5, started ~04:09; then u6 s2, u7 tau5 s0-2). gcn_k.py u1b s2 belongs to v21gfeat - not mine.
## Log
- 04:20 seed scaling (ana.py a, rank blend, e-5 best w, base/stackD): C25 7s +74.5/+83.2, 12s +85.3/+93.9; C23 7s +65.3/+76.4, 12s +76.8/+89.8 => 12 seeds = +11/+11 (C25), +11.5/+13 (C23): below +20 bar by itself, still rising. 1-seed C25: u1 d64L3 +35, u3 d128L3 +28 (standalone .7722 vs .7698), u4 L2 3s +28 vs u1 3s +57 => L2 worse, d128 no better.
- 04:25 (c) item-item co-click (ii.py; W=7d/all; standalone .59 C25/.58 C23, rho_M .10): adding to sum3 HURTS (C25 -5..-19e-5, C23 -3..-13e-5 vs baseline) -> DEAD.
- 04:30 (e) score combos (combo.py/combo2.py): weighting s_uv x2 vs cos_mean x1, s_h 0 ("sum3b"=2z(s_uv)+z(cos_mean)) beats sum3 (1,1,1): 7 seeds C25 +93.6/+102.0 (base/stackD best w.15) vs +74.5/+83.2; C23 +74.1/+85.0 vs +65.3/+76.4. OUT-OF-SAMPLE C26 (vs M, 7 seeds): sum3b +92.2@.15 vs sum3 +70.8 (+21e-5) => combo choice holds. 
  12 seeds + sum3b: C25 base 107.2@.2 stackD 117.4@.2 (+32.7/+34.2 vs 7s sum3 baseline); C23 base 87.3@.15 stackD 100.2@.15 (+22.0/+23.8) => PASSES bar on both cuts/bases (C23 margin thin). Needs 12 seeds on k6,k7 for eval (only 7 on disk).
- 04:32 DEPTH: u6 L4 d64 2 seeds C25: +72.6/+80.9 (350st) vs u1 L3 2 seeds +46.6/+52.7 => L4 ~ +26e-5 better at equal seeds (= L3 7 seeds). Testing L5 (u8) / L6 (u9) 2 seeds on k5 (jobs4.txt). Killed old runq2 queue (u6 s2 + u7 tau5 s0 still finishing; u7 s1,s2 dropped).
- 04:40 DEPTH (C25 1 seed, base/stackD best e-5): L3 +35/+40 ; L4 +60/+66 ; L5 +75/+84 (350-500st, standalone .792) ; L4 3s +82/+90 vs L3 3s +57/+64 ; tau-decay edges (u7) HURTS (+25/+30, standalone .764) -> dropped. Queued L6 (u9) , L8 (u10), L12 (u11) s0 on k5 (jobs4/jobs5).
- 04:50 DEPTH CONT. (C25 base/stackD best e-5; "sum3"): 1 seed: L6 350st +86/+94, 500st +91/+99; L8 350 +86/+92, 500 +101/+111 (standalone .7987); 2 seeds: L5 500 +88/+97, L6 500 +101/+111. 7-seed L3 baseline = +74.5/+83.2, 12-seed L3 = +85/+94. => ONE seed of L8 beats 12 seeds of L3. Deeper+longer keeps helping. L12 (u11 s0) running; k3 L8 800st (u10b s0) started to check C23. 
- 05:00 L12 (u11) k5 s0: 350st .765 (undertrained), 500st +80/+87 (rising, slower converge). k3 L8 (u10b s0, 800st run): C23 500st +89.7/+101.3, 650st +90.8/+102.7, 800st +86.9/+99.2 (7s L3 baseline +65.3/+76.4; 1s L3 +27/+31) => 1 seed L8 passes the bar on C23 (+24/+25) and C25 (+27/+27.5). Next: step-count test L8/L12 900st on k5 (u12/u13), then lock config, run seeds on k3,k5,k6,k7.
- 05:10 resumed (2nd interrupt). Step test k5 s0 (sum3 base/stackD e-5): L8 500st +101/+111, 650st +108/+118 (PEAK), 800 +101/+112, 900 +98/+108; L12 650 +102/+111, 800 +107/+116, 900 +108/+117 (flat; no better than L8) => LOCK config u12 = L8 d64 l2 3e-3 lr1e-2 650 steps (k3 650st was +90.8/+102.7; k3 emb_u12_k3_s0 symlink to u10b). 
  Parent correction: priority = low rho vs EN on eval + local gain. Running jobs8.txt (P2, 4 threads): k7+k6 s0, then seeds 1-5 on k7,k5,k3,k6. k7 L8 s0 gives first rho-vs-EN.

## 05:15 L8 (u12) first results (1 seed; gains e-5, base/stackD)
- rho vs EN on eval (200k sample, pearson of ranks; evalrho.py): old 7s L3 sum3 .9054 (reproduced), sum3b .9075, s_uv .893, s_h .890, cos_mean .897.
  L8 1 seed (k7 s0): sum3 .9089, sum3b .9100, uv+cm .9115, s_uv .8931, s_h .8915, cos_mean .8982; rho vs old G7 .976 (sum3). => L8 NOT less correlated with EN; ~.91 for any combo.
- Local gain L8 1 seed best-w ~.15-.2: sum3 C25 +105/+113 (w.15), C23 +91/+103; sum3b(2,0,1) C25 +122/+130 (w.15), +129/+139 (w.2); C23 +97/+109 (w.15). Passes bar (>+20 over 7s L3 on both) already at 1 seed. Optimal w moves up to ~.15-.2 (was .1).
- jobs8 still running (seeds 1-5). Next: fold/eval arrays, C26 vs M, fitg with G2.

## 05:17 tooling + 1-seed fit (L8 s0)
- final2.py <seeds> <name>: saves out/graph2{s3|s3b}_{name}_c{23,25,26}.npy (float32 v21seq val order), _eval_v21seqorder.npy, _eval_merged_rank.npy, graph2_eval_rowid.npy; prints table. evalrho.py = rho helpers. fitg3.py <G8 merged rank npy> <dloc> : 12-comp binormal fit (G7 old graph + G8 new graph; obs Q2,R1,S1,T1,G1,G2; G8 AUC prior = G7 posterior + dloc +-.002; rho from eval).
- 1 seed L8 (t1): C26 vs M s3b best +108@w.2 (L3 7s sum3: +71@.15); C26 standalone s3 .8077 / s3b .8078 (L3 7s .8056).
- fitg3 on t1 s3b (dloc .0021): fit G2 pred vs S1 +2.8e-5 (obs +1.2e-5). G8 replacing G7: best w8~.06, gain vs G2 +4.8e-5 (5/95 -1.8e-4..+2.6e-4), P>0 .64, P>5e-5 .50 => BELOW build bar (P>.75). On top of G2: ~0. Needs more seeds -> higher AUC/ lower noise.

## 05:55 resumed (3rd interrupt). u12 seeds 0-3 done on all cuts; seed 4 k7/k5 running (jobs8 continues to seed 5; don't duplicate).
- 2 seeds (t2, s3b, gains e-5 base/stackD at best w): C23 +108.7/+120.3 (w.15), C25 +135.4/+145.3 (w.2), C26 vsM +124@.2; standalone C23 .7724 C25 .7999 C26 .8096; rho EN eval .9144 (s3) .9128; rho G7 .971. Baseline 7s L3: C23 +65/+76, C25 +75/+83, C26 +71.

## 05:56 4 seeds (t4, s3b)
- standalone C23 .7742 C25 .8009 C26 .8104; rhoM .841/.874/.889; gains base/stackD best w: C23 +113/+127 (w.15-.2), C25 +139/+149 (w.2), C26 vsM +129@.2. rho EN eval .9165 (s3 .9146), rho G7 .974.
- fitg3 t4 (dloc .0048): G8 replaces G7: w8 .05 vs G2 +9.4e-5 (5/95 -9.5e-5..+2.7e-4) P>0 .80 P>5e-5 .66; w8 .07 +11.6e-5 P>0 .78; w8 .10 +12.4e-5 P>0 .72; on top of G2 (w7 .05,w8 .05) +4.2e-5 P .65. Passes bar IF dloc prior holds; sensitivity runs follow. Waiting for 6 seeds (seed 4 k7/k5 running, seed 5 after).
- 05:57 SENSITIVITY (t4): result hinges on dloc (G8 public AUC - G7 public AUC prior): dloc .0048 (=C26 diff) w8 .05 vs G2 +9.4e-5 P .80; dloc .003 -> +1.4e-5 P .55; dloc .0015 -> -5.3e-5 P .31. G7's own local->public transfer: C26 .8056 -> public post .8058 (1 datapoint, supports ~0 offset).
- 06:22 6 seeds f6 s3b: C23 +114/+129 (w.15-.2) C25 +140/+150 (w.2) C26 vsM +128@.2; standalone .7748/.8012/.8105; rhoM .844/.876/.890; rhoEN eval .9173 (s3 .9153), rho G7 .975. saturated vs 4 seeds.
- 06:25 fitg3 f6 (dloc .0049): G8 replaces G7 w8 .05 vs G2 +8.9e-5 (5/95 -9.9e-5..+2.7e-4) P>0 .79 P>5e-5 .64; w8 .07 +10.9e-5 P .76; w8 .10 +11.5e-5 P .70; on top of G2 (.05+.05) +3.7e-5 P .63. dloc .0035 -> +2.7e-5 P .59 (fails). Central assumption passes bar -> BUILT ONE: out/probe_v21graph_G3_S1_p50graphL8s3b6s.csv.zip = nr(S1+.05*nr(G8)), G8 = L8 6-seed s3b eval (out/graph2s3b_f6_eval_*). sha256 cc97c7615d8a17447acfa253913af67c8a60f0963fa2e4871463f1e94ca72e26 (14698290 B); ids==S1 order, no nulls; rho vs G2 file .99994. (SHA file had only G3 entries; an earlier identical-content build had different zip-mtime sha, superseded.)
- 06:30 copied to Mac /Users/erictao/Downloads/huawei/ ; shasum -a 256 on Mac == cc97c761...2e26 (OK). Not uploaded. No jobs running (jobs8 done).
