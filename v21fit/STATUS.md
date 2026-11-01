# v21fit STATUS (Eric: "Keep fitting on 0.8288, we need hit 0.831")  — updated 2026-11-01 ~19:50 Dublin
Base line: EB-new rank (ErBrown latest) public 0.828856, file /workspace/huawei/erbrown/w10an/eb_new_rank.npy. Target 0.831 (= +0.0022).
Never uploaded anything. No eval labels used. Anything using EB pseudo-labels => name has EBDISTILL; eval-history graphs => TRANSDUCTIVE/B.

## Known public scores used for fitting (from journal / build_log / probe READMEs)
- EB-old (ErBrown prior) / EB-new 0.828856 (pure EB-new rank recovered as (W-0.1*O)/0.9, W=submission_blend_newbest_v21_w10, O=rank of raw v21list B)
- EB-old + v21list B at w=.1 / .3 : actual +0.000405 / +0.000586 (w90/w70 files)
- EB-new + v21list B at w=.1: actual +0.000032 (the w10 file)
- p33 blends on v12/v12x: +0.000255 at w=.2, -0.000172 at w=.3  => p33 AUC ~ .8052
- Our own models ~0.819 public.

## Diagnostics (EB-new vs ours), approach 1
- CORRECTION to earlier note: the EB-old -> EB-new change is mostly WITHIN-slate (within var .00205 vs between-slate var .00045), not slate-level shifts.
- Rank corr: rho(EN,O)=.956, rho(EN,P)=.909 (P = p33).
- Binormal model (d'=sqrt2*PhiInv(AUC); combined d'=sum w d/sqrt(w'Rw)) reproduces known blend gains (E0+O w.1/.3: model +.00036/+.00048 vs actual +.000405/+.000586; EN+O w.1: model -.00002 vs actual +.000032).
- Predicted gains on EN: O (.819, rho .956): <= +.00004 at w~.065; p33: <=0 for all w>0; extrapolating away from E0: <= +.000025; 3-way EN -.129*E0 +.087*O: ~ +.0001.
- To get +.0022 an added model needs (AUC,rho to EN) ~ (.8295,.97) (.827,.95) (.8225,.92) (.819,.90) (.815,.88). Ours is (.819,.956). => 0.831 is NOT reachable by reweighting/blending of existing signals; realistic gain ~ +.00003 .. +.0001.

## Interim conclusion (after diagnostics + approach 1)
Blending/reweighting cannot reach .831. Remaining hope is approaches 2 (distillation: domain adaptation to stale eval rows) and a genuinely new signal. Distillation theory: student on EN soft targets -> EN (rho->1), gain ~0; sim running.

## Approach 2 simulation (running)
Teacher = .4z+.6L (C25 .81894, C23 .79865) shifted to mean p .0632; student ctl6 (noseq, ep6, alpha .3) trained on train + val rows w/ teacher soft labels. dist21.py, logs/dsim_A_c{25,23}.out, ens/.

### Approach 2 sim RESULT (2 seeds each; teacher T=.4z+.6L)
| cut | teacher T | student S (alone) | plain ctl6 (same 2 seeds) | rho(S,T) | rho(plain,T) | T+S best w | T+plain best w | .4z+.6M(8s) |
|---|---|---|---|---|---|---|---|---|
| C23 | .79865 | .79870 | .79845 | .9970 | .9943 | +.00021 (w.5) | +.00024 (w.5) | +.00031 |
| C25 | .81894 | .81907 | .81900 | .9981 | .9971 | +.00019 (w.7) | +.00020 (w.5) | +.00025 |
=> Distilled student reproduces the teacher (rho .997-.998) and alone is +.00005/+.00013 over T; adding it to T gives no more than a plain ctl6 added to T. NULL result: distillation does not create information beyond teacher+plain. For EN (.8288 teacher), student alone ~EN+small; EN+S gain expected <= +.00005.
Teacher for C27 = EN rank -> logit via sorted (.4z+.6M ours) logits, shifted to mean p .0632 (teach_EN_c27.npy). EBDISTILL run launched (tag EBDISTILL, seeds 0-2) -> v21fit/ens/EBDISTILL_A_c27_s*.npy.

### Approach 5 (B = TRANSDUCTIVE, ctl8B) validation  (8 seeds ctl6; full table in B_val.md)
| cut | A ctl6 8s | B ctl6 8s | A blend .4z+.6M | B blend | A-base |
|---|---|---|---|---|---|
| C23 | .79862 | .79865 | .79903 | .79892 (z_B weaker .79647 vs .79679) | |
| C25 | .81904 | .81906 | .81921 | .81923 | |
| C26 | .82797 | .82796 | - | - | |
=> B == A within noise. No B-based EB blends warranted; keep A (train-only).

### Approach 3 (two-level decomposition) — NULL  (dec.py)
- Slate-level is almost irrelevant for AUC: slate-mean-only AUC ~.52 (C23 z .5228/M .5161; C25 .5265/.5212), within-slate-only ~.789-.810.
- Rank corr, between/within: C23 z-vs-M .9315/.9826; C25 .9557/.9943; EVAL EN-vs-O .9254/.9588; EN-vs-E0 .9614/.9868; EN-vs-P .8034/.9171 => EN differs from ours mostly in WITHIN-slate order.
- Shift of blend's slate means toward M's slate means (lam .25/.5/.75/1): C23 -.00001/-.00008/-.0002/-.00038; C25 +.00005/+.00004/0/-.00009. Null. No decomposition probe worth uploading.

### Approach 4 (segment weights) — NULL  (seg.py)
z/M blend weight per segment (hist_n_all, tm_stale, us_since_ev): in-sample gain <= +.00005; cross-applied C23->C25 / C25->C23: hist -.00003/+.00003, stale 0/+.00003, since -.00006/0. Not worth it.

### Public-score fit (opt.py, binormal; rank corr on eval): rho(EN,E0)=.9849 rho(EN,O)=.9563 rho(EN,A_all)=.9562 rho(EN,B_all)=.9560 rho(EN,P)=.9092 rho(O,A)=.9992 rho(A,B)=.9996. Assumed AUCs: EN .828856 E0 .824558 O .8190 A .8197 B .8195 P .8052.
Optimal weights / predicted gain over EN: EN+A w.082: +.000059; EN+O w.043: +.000018; EN-.136E0+.098A: +.000124; EN-.115E0+.061O: +.000066; +P(-.056): +.000173 (speculative, not used).
=> A_all ~ same as O in rank; EN+.1*O already scored +.000032 public. Best new probe = 3-way EN-.136E0+.098A (predicted +.0001).
EBDISTILL: seed0 student rho(S,EN)=.965 (A_all alone: .957): student moved only .008 toward EN (EN info is not learnable from our features, unlike sim where teacher was a function of the same features). Break-even AUC for S at rho .965 is .8203 (needs >.8203 to help).
Running now: EBDISTILL seed2 (v21fit/ens/EBDISTILL_A_c27_s2.npy), finishes ~20:40.

## FINAL (20:40 Dublin) — all background jobs finished (none running)
EBDISTILL 3 seeds done: rho(S,EN)=.9654, rho(S,A_all)=.9952.
Probes in v21fit/out (529,390 rows, rowid order = merged.parquet = w10 submission, no NaN, [0,1]); copied to Mac ~/Downloads/huawei, shasum -c OK:
47a7abe270dfcc697e77d79abaef2929d58578d306a9321e5da2a5b225516084  probe_v21fit_P1_EBnew_m136E0_p098A.csv.zip
a1d94ba2cdd82b7ec9521383b5edf3dff105a705b4aa3b582369f20602f551f0  probe_v21fit_P2_EBDISTILL_EBnew90_S10.csv.zip
7e043b34e892ae09f58e5582552555acf5debe61cd7f6e582e8272981e7e9179  probe_v21fit_P3_EBDISTILL_EBnew_m136E0_p050A_p050S.csv.zip
Upload order & expected public delta vs .828856: P1 +.00005..+.00012 (could be slightly negative); P3 ~+.00005; P2 ~0..+.00005 (EBDISTILL). 0.831 not reachable by these.

## Track 1/2 round (from 20:45 Dublin)
- Inventory rho with EN on eval (inv.py): v21slateB .9541, v21slateA .9543, v21listA .9566, A_all .9562, p33blend70 .9511, probe_blend_p3 .9394, v8regen .9486, v5 .8708, v5c .8832, p3_d16 .8174 (public .78 => below break-even AUC .782 at that rho, useless), E0 .9850, w10 .9996. v21slate/list/seq family rho(A) .996-.9995 => one signal, no new info. submission_lgbm (.7356) file not available (and far below break-even). => Track 2 inventory adds nothing beyond P1-P3.
- rec2 (ctl6 trained on last 2 train folds only, C25 s0): AUC .81841 (ctl6 s0 .81890), rho(M0) .9947, blend gain into .4z+.6M8 (.81921): w.2 .81926 (+.00005), w.35 .81923, w.5 .81914. = just another seed. NULL.
- dcn3 (DCN-v2 cross x3 + deep, C25 s0, lr2e-3, 6ep): overfits (ep0-5 .81733 .81743 .81729 .81414 .80212 .78558). ep1 snapshot: AUC .81743, rho(M0) .9862 (more diverse), blend into .4z+.6M8 w.2 .81929 (+.00008), w.35 .81920. Small.
- wide1 (hid1024 drop.35 noslate, C25 s0): AUC .81886, rho(M0) .9978, w.2 .81922 (+.00001). NULL.
- Launched dcn4 (DCN regularised lr7e-4 drop.4 edrop.3, 4ep, cores 0-3) and dcn5 (same, cl2, last 2 folds, cores 4-7), C25 s0.
- dcn4 (regularised DCN lr7e-4 drop.4 edrop.3, ep snapshot 2 of 4: AUC .81812 [ep0-3 .81730 .81800 .81812 .81569]), rho(M0) .9873, rho(z) .9839: blend into .4z+.6M8 (.81921): w.2 .81944, w.35 .81946 (+.00025), w.5 .81936 => genuine diversity, same size as whole 8-seed ctl6 gain. dcn5 (cl2,2 folds) weak (.81745 ep1, +.00009). Next: dcn6/dcn7 with ep=3 (schedule completes).
- dcn6 (regularised DCN, ep3 full schedule): AUC .81827, rho(M0) .9891, w.35 .81943 (+.00022). dcn7 (cl4 drop.5 lr5e-4, snapshot ep1): .81825, w.35 .81934. Average of dcn4/6/7 (D3): AUC .8189, rho(M8) .9922; best .3z+.3M+.4D = .81949 (+.00028 over base .81921, local C25). Decision: build C27 DCN eval preds (dcn6 cfg s0 cores 0-3; dcn7 cfg s0 cores 4-7) -> probe with A_all mixed with D. Expected EN gain small (~+.00005).
- 21:30: C27 DCN: dcn6 s0 + dcn7 s0 done (ens/dcn6_A_c27_s0, dcn7_A_c27_s0); running now: dcn4 cfg s0 (cores 0-3) and dcn6b seed1 (cores 4-7), ~6 min each. mkD.py builds D_dcn.npy (args tag:col:seed).

## TRACK 2 FINAL + DCN (21:38 Dublin)
C27 DCN eval members: dcn6 s0 (ep2), dcn6b s1 (ep2), dcn7 s0 (ep1), dcn4 s0 (ep2) -> D_dcn.npy. rho(D,A)=.9935, rho(D,EN)=.9503 (A: .9562), rho(D,E0)=.952.
Local C25: D3 .8189 alone, .3z+.3M+.4D .81949 vs base .81921 (+.00028).
Binormal (opt2.py, A=.8197): D AUC assumed .8192 => EN+.109D +.00011; EN-.136E0+.098*(.5A+.5D) +.00016; EN-.136E0+.098D +.00018. If D AUC only .8180: +.00010 / +.00006. P1 (A only) +.00012.
Probes (out/, appended to SHA256SUMS_v21fit.txt, on Mac Downloads/huawei, shasum -c OK for all 5):
7842c18d9d7eb2b6c79036c7ea716b6022c8d0162029aff72add02632bd22c0e probe_v21fit_Q1_EBnew_m136E0_p098_A50_DCN50.csv.zip (rho EN .99947)
2f86d30fb3a8f030d5d8661c47c946e0f282858c1b26069af869af9722646941 probe_v21fit_Q2_EBnew_m136E0_p098DCN.csv.zip (rho EN .99941)
Upload order: Q1, P1, Q2 (P1 sha 47a7abe2...6084 already on Mac). Stray duplicate of Q1 zip at /Users/erictao/ (first copy landed in home) - not deleted.
No jobs running.

## Round 3 (21:46+ Dublin; cores 6-7, RSS<0.5GB) — v21dcn worker: /workspace/huawei/v21dcn did not exist at 21:50
1. Sensitivity (sens.py, 4000 MC draws: A AUC U(.8185,.8205), D=A-U(0,.0025), pairwise rho +-.003 N, model miscalibration N(0,.00005)):
   predicted gain over EN (.828856): P1 mean +.000103 [5%..95% -.000073..+.000284] P(>0)=.83; Q1 +.000109 [-.000049..+.000273] P=.86; Q2 +.000106 [-.000104..+.000319] P=.80.
   Public-subset sampling noise of the diff (pubnoise.py, local C25 analog rho .9995): SD .00004 (full eval) .00007 (30% subset) -> small vs model uncertainty.
2. Other forms (forms.py, C25 .3z+.3M+.4D): rank .81949, zscore-logit .81949, probit-rank .81950, rank^2 .81950, rank^.5 .81949, sigmoid prob .81946 => all equal (+-.00001). NULL.
   Negative weight on p33 (sensP.py): sign/size depends on its assumed public AUC (.800 -> opt wP -.134 gain +.0005; .8052 -> -.039, +.00004; .810 -> +.07): unsupported, skipped. v5 has no known public score -> skipped.
- 22:00: v21dcn worker running r1 screening on C23/C25 only (cores 0-5); no C27 eval array yet. Polling its out/ and ens/*c27* every ~20 min (until ~00:45).
