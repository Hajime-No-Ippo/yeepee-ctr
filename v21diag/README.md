(Renamed from v20diag -> v21diag on 30 Oct 2026 to avoid clash with Code Buddy's v20; all outputs labelled v21.)
# v21diag — diagnostics + 3 experiments on top of v19 (copies only; originals untouched)
Metric global ROC AUC. B = TRANSDUCTIVE, A = train-only. Nothing uploaded. Scripts: diag1.py (step 1), sim_headroom.py. Raw numbers: logs/diag1.json, logs/diag1.out, logs/sim_headroom.out.

## STEP 1 — error breakdown (v19 saved preds; blend=(dwbig+dw_s+2*mean MLP x3)/4, reproduces .81830 B / .81829 A at C25; C23 only dwbig exists, no blend)
Rows are event-major 10-blocks (checked eid). C25: 43,964 slates, 6.335% CTR, 63.35% slates with a click. C23: 50,155 slates, 62.92% with a click.
| | C25 B blend | C25 A blend | C25 B dwbig | C23 B dwbig | C23 A dwbig |
|---|---|---|---|---|---|
| global AUC | .81830 | .81829 | .81742 | .79647 | .79679 |
| within-slate AUC (mean rank of the click among 10) | .80705 | .80711 | .80605 | .78505 | .78527 |
| slate-has-click AUC: max / mean / sum p / lse / 1-prod(1-p) | .623/.571/.670/.667/.672 | .624/.571/.670/.668/.672 | .623/.572/.670/.666/.669 | .604/.566/.648/.642/.645 | .605/.566/.648/.643/.646 |
| global AUC using only softmax within-slate share | .81215 | .81214 | .81130 | .79078 | .79093 |
| global AUC using only slate-level sum p | .56654 | .56660 | .56638 | .55838 | .55875 |
| ORACLE within (click swapped to slate's max score) | .96655 (+.14825) | .96658 | .96721 | .96464 (+.16817) | .96469 |
| ORACLE across (clicked slates shifted above no-click slates, within order kept) | .89143 (+.07313) | .89140 | .89106 | .87940 (+.08293) | .87961 |
| both oracles | .98625 | .98626 | .98679 | .98595 | .98596 |
| across shift +0.5 / +1.0 / +2.0 logit to clicked slates (partial oracle) | .84452/.86331/.88322 | | .84382/.86274/.88280 | .82527/.84635/.86943 | |
Pair accounting: pos-vs-neg pairs inside the same slate are only 0.002% of all pairs (9/N_neg); global AUC is therefore ~100% "across-slate" pairs, but those are decided by BOTH slate-level offset and within-slate order (a clicked row ranks above negs of other slates by being high in its own slate AND by being in a hot slate). Softmax share alone gives .81215, adding slate-level information (sum p) lifts to .81830 (+.00615) — the model already uses slate-level info; slate AUC .67 is the cap observed (v7: .67, DW: .65-.66).
Headroom: within-slate +.148 (C25) / +.168 (C23); across-slate +.073 / +.083. Within is ~2/3 of the gap, across ~1/3 (same as v7 finding).
Sensitivity (sim_headroom.py, C25 B blend): inject independent Gaussian evidence about "slate has a click" into prior P(has)=1-Prod(1-p), p_row=P_post*softmax_share. Row prob = prior*share alone = .81700 (vs blend .81830 — product form loses .0013 so any recalibration must be a correction ON TOP of the blend, not a replacement).
| extra evidence (alone slate AUC) | slate AUC of posterior (delta) | global delta |
|---|---|---|
| d=.1 (.53) | +.0018 | +.00015 |
| d=.2 (.56) | +.0054 | +.00054 |
| d=.3 (.585) | +.0146 | +.00121 |
| d=.5 (.64) | +.0400 | +.00340 |
=> roughly +.0008..+.0009 global per +.01 slate AUC. +0.002 from slate-level alone needs slate AUC ~.667 -> ~.69, i.e. independent evidence about whether a slate gets any click that is itself ~.58-.60 AUC. Prior probes (v7/DW) found event-level features capped near .61-.62 from user stats; lse of the model gives .667 already.

### Segments (B blend C25; rows share / CTR / global AUC in segment / within-slate AUC / weighted deficit share*(overall-seg))
History length (us_hlen): 0 (0.8%): AUC .564 within .534 CTR 4.5% | 1-2 (3.7%) .729/.722 | 3-5 (9.6%) .761/.752 | 6-10 (18.6%) .800/.792 | 11-19 (25.1%) .827/.819 | 20 (42.2%) .838/.828
User activity (clicks to date) deciles d0..d9 (7.3/10.5/11.3/10.3/8.9/11.6/8.6/10.0/11.4/10.2%): .726/.774/.800/.821/.830/.827/.831/.838/.839/.843 ; within .712/.765/.791/.813/.820/.821/.824/.829/.827/.832
New vs returning: us_cold (no train rows) 3.3% .744 (within .721) vs warm 96.7% .820; no-history 0.8% AUC .564 (CTR 4.5%) 
Item popularity (it_imp deciles, share ~10% each): .773/.792/.803/.816/.821/.822/.829/.825/.838/.840
Time of day (UTC bucket 0..3): .824/.824/.814/.811 (21.7/28.6/26.9/22.9%; within .814/.813/.802/.799)
Staleness (event ts − cutoff): <6h 18.6% .827 | 6-24h 61.5% .820 | >=24h 19.9% .804
In-history candidate (hist_in=1): 2.6% of rows, CTR 30.1%, AUC .681 (!) vs not-in-history .808.
Position in slate: no position column; row order inside slate carries no signal: slot-ordinal single AUC .498 (C25) / .499 (C23), CTR by slot .0614–.0657 (noise). Step 5 therefore not run (already established null: research/data_dig_v1.md F3; user also excluded it).
Weighted deficits (C25): low activity d0-d2 ≈ .0135; hlen<=10 ≈ .0142; cold users .0025; no-history .0021; staleness>=24h .0028; hist_in .0035 (rows where the candidate is in the user's history, AUC only .68); item tail d0-d2 .0084.
C23 (4 days, harder, staleness up to 4d): AUC by staleness <6h .812, 6-24h .804, 1-2d .778, 2-3d .772, 3-4d .763; eval is 33% >1 day stale (up to 7 d; Monday 66.6%) vs val25 20%, so eval sits between C25 and C23 in difficulty (earlier work: v7_error_analysis.md).

## STEP 2 — final-fit data/rounds audit (code: v19/xlgb17.py, ens17.py, mlp17.py; v19/run_blend19.sh)
- Training folds are f_k = FIRST event per user in [C_k, C_k+1d), features built at cutoff C_k (no later info). Rows: f1 351,560 / f2 351,120 / f3 349,120 / f4 350,400 / f5 352,300 / f6 351,380 (~35k events each). The train file has 686,824 events (~98k/day), so folds use ~36% of events.
- Fold window is a SLIDING 4-fold window: range(max(1,cut-24), cut-20): C23 = f1..f2, C25 = f1..f4 (n=1,402,200), C26 = f2..f5, C27 (final, predicts eval) = f3..f6 (n=1,403,200). So the final fit does NOT use all labelled data: f1,f2 (Jan 21,22 targets) are dropped, and the other ~64% of events (non-first events; e/d/E folds) are unused. Final (C27) therefore has the same amount of data as C25 by design (identical recipe, so local C25 numbers mirror the final).
- Rounds/epochs are fixed, not scaled: dwbig 1000 rounds (lr .015, 127 leaves) saved only at 1000; dw_s 500 rounds (lr .03, negfrac .5 -> n=745,540; blend uses 500 col); MLP 4 epochs OneCycle (negfrac .5), blend uses ep4 (col 3).
- Evidence on more data (existing logs, no new run): v9d S3 (old v8-lite pipeline, blend delta): nfolds 4->3 −.00019/−.00020 (c25/c26), 4->2 −.00044/−.00033, c23 1 fold −.00136, **nf5 vs nf4 at c26: −.00002** (adding a 5th fold: no gain). DW learning curve (dw_lb/findings.md #9, 1->4 folds at C25: .8142/.8158/.8162/.8160) saturated after 2–3 folds; v8 screens: adding e1..e4/d1..d5 folds (non-first events, +2.8M rows at C27, fit 2.3–2.6x) = +.00020/+.00030 on GBDT blend, +.00013/+.00020 with seq features. C23->C25 is not a clean proxy (different val sets: staleness up to 4d vs 2d), so I used the same-set fold-count ablations above.
- Estimate: using f1..f6 (6 folds) at C27 ≈ 0..+.0001; adding the extra e/d folds ≈ +.0001..+.0003 at 2.5x fit time (breaks the runtime budget). More rounds/epochs per data amount not tested (no saved round curve past 1000; dwbig saved only at 1000). Verdict: NOT a +.0005–.001 free lunch. Cheap test if wanted: dwbig C25 trained on f1..f4 vs only f2..f4 at 1000 rounds with save curve 600/800/1000/1200/1500 (~6 min each).
- Possibly more valuable: train/eval staleness mismatch (train folds are all <1 day stale, eval 33% >1 day, up to 7d; seq-feature gain by staleness was ok per v9d) — see recommendation.

## Log of experiments below (steps 3,4) — appended as they finish

## STEP 3 — across-slate calibration (DONE)  [logged 2026-10-30 ~07:15 Europe/Dublin]
Score = z - c*lse(z) + a*P, P = slate P(click) model (LightGBM, trained out-of-time, step3.py), (a,c) tuned on C23 dwbig -> a=.3, c=.5, tested on C25.
- C25 B blend: .81830 -> .81873 (+.00044; event-bootstrap sd .00007, P(>0)=1.00). C25 A blend: .81829 -> .81869 (+.00040, sd .00007).
- No C23 hurt (params tuned there). Grid is a narrow ridge (c~.5,a~.2-.3; c=.25,a=.1-.2 +.0003); off-ridge hurts down to -.004.
- Segments B: <1d +.00037, >=1d +.00068, with-history +.00043, empty-hist +.00104 (A empty-hist -.0029, caveat).
- Candidates (out/, copied to Mac ~/Downloads/huawei, shasum -c OK):
  submission_v21slate_B_TRANSDUCTIVE.csv.zip 07949d2df6bcaca6e634bb230898fe305afb3c0b7ffede18cc76eebc0ef0dc58
  probe_v21slate_B_TRANSDUCTIVE_p33_w80.csv.zip acbebeac5f42f1555fceda8212d4945ec7508bd57fbc22a488983323799ba4eb
  submission_v21slate_A_trainonly.csv.zip ef4d73da90b254e17857b84f8d9ef9561d2eb210998c9056d32ebea0d7b82664
  probe_v21slate_A_trainonly_p33_w80.csv.zip c327e57af99e6ce84909b8a432a37226ce702acbf2919f6db65bc162263017f4
  (uncompressed CSVs deleted from out/ to save space; regenerate with mk21.py)

## STEP 4 — generator fit (NOT RUN; resume notes)
mlp21list.py is written (listwise slate softmax w/ outside option; copy of v19/mlp17.py; syntax-checked only). Box was busy (load ~4, Code Buddy s0_ref.py running) so heavy run was not started.
Resume: /workspace/venv_v17/bin/python mlp21list.py 25 v21list --ep 4 --seed 0 --loss list --lr feat_lr2,feat_tt,feat_pp,feat_sqC,feat_dw,feat_Bp,feat_NBs  (also --loss bce baseline, and cut 23), when pgrep idle; then test as blend member and stacked with step-3 correction.
STEP 5 (position): excluded by owner; slot AUC .4979/.4990 (null).

## STEP 4 results so far [2026-10-30 ~08:00 Europe/Dublin]  (mlp21list.py: listwise slate softmax + outside option, 1 seed, 4 ep; OOM fixed by in-place transform)
Standalone (global / within / slate sum_p AUC):
- C25 B list-MLP s0: .81854 / .80691 / .6790  (v19 blend: .8183/.80705/.6701; dwbig .81742)
- C23 B list-MLP s0: .79798 / .78607 / .6567  (dwbig .79647/.78505/.6475)
Blend (logit-converted list score; w = weight of list in (1-w)*z_v19 + w*list):
- C25 on v19 blend: w.25 .81873, w.5 .81892, w.75 .81886 ; with slate corr (.3,.5): .81893/.81896/.81880 ; v21slate = .81873
- C23 on dwbig (v19 blend preds not stored at C23): w.25 .79773, .5 .79842, .75 .79850; dwbig .79647; v21slate-dwbig .79691; corr on top adds ~0 (list absorbs it).

## STEP 4 FINAL [2026-10-30 ~10:00 Europe/Dublin]
List-MLP (mlp21list.py, listwise softmax + outside option), B variant, 3 seeds, ep3 (final) logits->logit space, blend z=(1-w)*v19blend + w*list.
- Standalone C25: s0 .81854, s1 .81844, 3-seed avg .81856 (within .80694, slate sum_p AUC .6792); BCE same arch .81739 (slate .6697). C23: s0 .79798, 3-seed .79818 (dwbig .79647).
- Blend w=.4/.5/.6/.7 C25 (v19 blend .8183; v21slate .81873): .81887/.81891/.81892/.81889. C23 (dwbig .79647; v21slate-on-dwbig .79691): .79827/.79849/.79862/.79865.
- Slate corr on top adds ~0 / negative at C23 (list absorbs it). Chosen (tuned C23): w=.6, no slate corr. C25 .81892 (+.00019 over v21slate), C23 +.00215 over dwbig (+.00171 over v21slate).
- A variant: list s0 C25 .81857 / C23 .79799; blend w=.6: C25 .81894 (A v21slate .81869, A v19 .81829), C23 .79865 (A dwbig .79679).
- L2-LR member (lr21.py, torch linear, same rank-gauss inputs): standalone C25 .81043, C23 .78943; adding to list-blend w=.05/.1/.2: C25 .81887/.81877/.81846 (base .81892), C23 .79858/.79849/.79818 (base .79862) -> NULL/negative.
- Staleness-matched training: NOT re-run; prior /workspace/huawei/staleness_test (27 Oct) found delay-matched rows B/B_w -.0001..-.0036 rw AUC on both folds. Skipped.
- Candidates (C27: 3 seeds each, sliding f3..f6): out/submission_v21list_{B_TRANSDUCTIVE,A_trainonly}.csv.zip, probe_..._p33_w80; sha in SHA256SUMS_v21list.txt; copied to Mac, shasum OK.
