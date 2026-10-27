# v11 notes (quality first; runtime recorded, not constrained). Times Europe/Dublin (UTC+0).
Harness: copy of v10 (cl.py D=v11, ens.py +--save rounds, e.sh runner 8 threads, cmp11.py NAME SPEC -> cmp11_results.jsonl with d8_ (vs v8 trim) and d10_ (vs v10B)).
Feature dirs symlinked read-only (feat_sq10 -> v10/feat_sq10). seq11.py = seq10.py + env knobs SQ_D/SQ_DH/SQ_H/SQ_HID/SQ_HL/SQ_SEED/SQ_TEMB.
Refs: v8 trim C25 .81321 C23 .79056 C26 .82188 | v10B C25 .81388 C23 .79102 (rw .78984) C26 .82305. Accept: +C25, C23 not < -0.0002.
## Checkpoints
- 17:55 b1.sh: LightGBM capacity (nf1 = negfrac 1.0 for b and s; big1 = lr .015 leaves127 ff.5 l2 30 maxbin255, 3000 rounds) on cuts 25,23.
- queued sw1.sh: raw seq-model sweep on val25 (base, ep5, d64, temb, hl50, np6, h4, seed1) -> sw/*
- 18:25 b1 results (single models, deltas vs v10B blend): nf1 (negfrac 1.0) b/s pair C25 +.00006 C23 +.00002 -> no gain (negfrac .5 fine).
  big1 (lr.015 l127 ff.5 l2 30 mb255) best @1000 rounds: C25 .81406 (+.00018) C23 .79092 (-.00009); overfits after 1000 (3000: .81313).
  big1@1000 + v10B b + v10B s (raw mean): C25 .81416 (+.00028) C23 .79134 (+.00033) -> small ensemble gain. fit 346s/276s (8 thr).
- 18:40 sw1 raw seq sweep (val25 sq_logit AUC, single J25 fit, 8 thr ~90s): base .78189 | ep5 .78013 | d64 .77837 | temb (trainable item emb) .77251 |
  hl50 .78216 | np6 .77649 | h4/hid128 .78250 | seed1 .78331 ; avg(base,seed1) .78483 (+.0022) -> seed noise dominates; multi-seed averaging is the lever.
- 18:45 msA: 4 seeds (1-4) full seq runs concurrently (2 thr each, ~2GB RSS each) -> msA/s{1..4}; avg_sq.py averages (+ feat_sq10 = seed0) -> feat_sqA
- 18:50 msA done (wall ~530s/seed at 2 thr, 4 concurrent; seed3 OOM-killed during eval -> eval rerun alone 80s; max 3 concurrent from now).
  raw sq_logit: single seeds C25 .7818-.7833 / C23 .735-.757; feat_sqA (avg feat_sq10+s1,s2,s4) C25 .78576 C23 .75637 C26 .79540. avg_sq.py adds sq_sd (seed sd).
- 19:05 b2 (LightGBM on feat_sqA, deltas vs v10B): sA b+s pair C25 .81416 (+.00028) C23 .79273 (+.00171) C26 .82328 (+.00022)
  sAbig@1000 C25 .81441 (+.00053) C23 .79276 (+.00174) | sAbig@1000+sA_s@500 C25 .81449 (+.00060) C23 .79307 (+.00205) | big+b+s .81446/.79308
  -> ACCEPTED: multi-seed seq + big LGB. Next: more seeds (msB 5-7 base, msC 8-10 h4) -> feat_sqB.
- 19:35 msB(5-7) msC(8-10, h4/hid128) done. feat_sqB (8 seeds) raw C25 .78692 C23 .76080 C26 .79570; feat_sqC (11 seeds) .78770/.76223/.79648. seq runtime ~540-680s/seed (3 concurrent, 2 thr).
  b3.sh: LightGBM sC_s + sCbig on feat_sqC, cuts 25 23 26.
- 20:20 b3 (feat_sqC, 11 seeds; deltas vs v10B / v8): sC_s C25 .81402 (+.00014) C23 .79281 (+.0018) C26 .82321 ; sCbig@1000 C25 .81460 (+.00072) C23 .79309 (+.00208) C26 .82340 (+.00035)
  sCbig+sC_s C25 .81459 (+.00071; +.00138 vs v8) C23 .79339 (+.00237) rw .79222 C26 .82355 (+.00049)  <- current best. Seed averaging saturating (8 seeds -> 11 seeds: +.0002 C25).
  Next: MLP second family (mlp11.py) rank-blend; then seq variants.
- 19:35 interruption; shipping current best FIRST: b4.sh = ens.py cut 27 for sC_s and sCbig on feat_sqC -> ens/*_c27_s0.npy

## CHECKPOINT 19:50 (Dublin) - SHIPPED candidate v11a
- Config: LightGBM sCbig@1000 (big params, negfrac 1.0, nfolds 4) + sC_s@500 on feat_lr2/tt/pp + feat_sqC (11-seed averaged seq), raw logit mean -> sigmoid. mksub.py builds out/submission_v11.csv.
- Local: C25 .81459 (+.00071 vs v10B, +.00138 vs v8), C23 .79339 (rw .79222), C26 .82355.
- Eval run (cut 27): sC_s fit 35s, sCbig fit 161s + pred 38s. Seq per seed ~540-680s at 3 concurrent / 2 threads, eval fit ~80s.
- submission_v11.csv: 529,390 unique rowids == v8 set, p in [0.00086,0.797], Spearman vs v10 .984, vs v8 .988. Gate refused (.79222<.83) as expected. out/submission_v11_candidate.csv.zip + .sha256 written.
- MLP2nd family (mlp11.py, feat_sqA): alone C25 .81437, C23 .79251, ~67s. Blend not yet evaluated. NaN fix applied in mlp11.py.

## CHECKPOINT 20:05 - SHIPPED v11b (LGB + MLP rank/logit blend)
- MLP on sqC (mlp2, 4 ep, 1 seed): alone C25 .81456 C23 .79302 C26 .82405 (~65s/cut).
- Blend sCbig:1+sC_s:1+2*mlp2:3 (logit mean, weights 1:1:2): C25 .81522 (+.00134 vs v10B), C23 .79417 (+.00315, rw .79300), C26 .82434 (+.00128). Weight 1x: .81515/.79412; 4x: .81514/.79400. ACCEPTED.
- out/submission_v11.csv = v11b (previous LGB-only version kept as out/submission_v11a_lgb_only.csv). NOTE: C25 has no 48h+ rows so rw25 == auc25; the ".79222" figure was C23's rw.

## FINAL 20:25 - v11c SHIPPED (3-seed MLP)
- Blend sCbig@1000 + sC_s@500 + 2*mlp2@ep4 (seeds 0,1,2), logit mean -> sigmoid. Local: C25 .81524, C23 .79424 (rw .79308), C26 ~.8243. vs v10B: +.00135 / +.00324. MLP seeds saturate (+.00002). 7-epoch MLP no gain.
- Eval: out/submission_v11.csv; zip out/submission_v11_candidate.csv.zip sha256 38e8bc45...5aeb0d (see .sha256). Gate refused (.81523<.83).
- Cleaned: sw/, feat_sqA/sqA5/sqB, old ens (nf1,big1,sA*,mlp1,mlp3). Kept: feat_sqC, msA/msB/msC seed outputs, ens for v10B/sC*/mlp2.
