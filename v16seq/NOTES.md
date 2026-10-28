# v16seq — next-click sequence models (SASRec, GRU4Rec) as feature generators
(times Europe/Dublin = UTC)
## Design
- seqlib.py: click log of train events with ts<C (histories + clicks, dedup by (user,item,minute); cut C_k = J20+k*DAY), per-user sequences, windows of 51 clicks (stride 25, right-anchored padding), input = item emb + log2-minute time-gap-to-NEXT-click emb (+ learned positions for SASRec).
  Inference: user state = last hidden state over [log(ts<C) + target event's own history (ts<event ts)], last 50 clicks, last token gap = event ts - last click. Also states from last-1 and last-3 clicks only.
- SASRec: 2 pre-LN blocks, 2 heads, d64, dropout .2, tied item emb. GRU4Rec: 1-layer GRU d64, tied emb. Loss: sampled softmax (K=256 uniform shared negatives + hit masking) or gBCE (t=.75).
- 10 features/model: dot, cos, z, gap-to-max, log-softmax over the 10, rank, dot_last1 (demeaned), cos_last1, dot_last3 (demeaned), cos_last3. NaN if no history at all.
- Strictly label/eval-free: no events >= C in sequences; eval rows only use their own history array.
## Checkpoint 1 (tuning on val26 with model C6; val26 not a reported cut)
raw global AUC of feature on val26 (dot / z): GRU+sampled-softmax 10 ep .5282/.5271 (124 s fit); SASRec sampled-softmax 10 ep .5285/.5278 (343 s); SASRec gBCE t=.75 10 ep .5134/.5138 (340 s); gBCE 2 ep .5016. => softmax CE used for both. For reference co_cos_last .5202, feat_sqC sq_logit .5020 (global AUC of v12 sq features are near .5 as well; the next-click models do NOT see slate CTR priors, unlike the DIN (.787)).
Runtime: GRU fit ~100-120 s/cutoff (8 thr), SASRec 8 ep ~4-5 min; seqfeat.py logs in timings.jsonl (gru s1 C3 847 s is polluted by contention with resid16.py: running 2 torch/LGB jobs concurrently = 6x slowdown; never do that).
Features built (val23/val25/val26): gru_s0,s1,s2 ; sas_s0 (running).
## Checkpoint 2 — correctness review (06:00 Dublin/UTC)
- Leakage: all table events have ts>=C (val23/25/26/eval: min ts-C=0), click log is ts<C, history items filtered ht<ts. Candidates all mapped (0 unknown, incl. eval). Row order asserted against feat_v3 rowid (e constant across 10 rows).
- Objective: uniform shared K=256 sampled softmax with accidental-hit mask (candidate-in-softmax, not in-batch); gBCE worse (.513 vs .529). Last-token gap = event ts - last click.
- Why raw AUC is weak: within-event HR@1 on val25 (random .1): gru_dot .1133, it_ctr .1185, co_cos_last .1377, mf_score .1488, sv_k64_ucos .2624. Slate candidates are driven by the impression/slate structure (what the platform showed), which next-click models trained on click sequences do not see; click sequence information is already covered by co_cos/sv/mf.
## Checkpoint 3 — residual on v12 logits (resid16.py oof, GroupKFold5 by user, LGB init_score=v12 z, 200 rounds; noise ±.0001)
d_vs_v12 / d_vs_base (new - base) for C25 | C23 | C26:
- gru_s0:                C25 +.00004/-.00000 | C23 +.00003/+.00005 | C26 -.00003/+.00008
- gru s0+s1+s2 (avg):    C25 +.00004/+.00000 | C23 +.00011/+.00012 | C26 +.00000/+.00011
- sas_s0:                C25 -.00000/-.00005 | C23 -.00004/-.00003 | C26 -.00001/+.00000(.0001)
- gru(3 seeds)+sas:      C25 +.00001/-.00003 | C23 +.00005/+.00006 | C26 +.00002/+.00013
Threshold was >= +.0005 on C25. Max observed +.00004 => **NO-GO**. Full retrain NOT run, no submission_v16seq built.
Runtimes: GRU 100-120 s per cutoff, SASRec ~4-5 min per cutoff (8 threads); residual run ~35 s per spec (3 cuts).
