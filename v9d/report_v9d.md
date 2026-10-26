# v9d public-gap screens (26 Oct 2026)
Lite v8 control (trim) reproduced exactly in v9d/val (c25 base 0.81097 / slow 0.81159).
dz8 = delta of 0.8z+0.2sq vs v8-lite; dz8_w = same, eval-likeness weighted (adv weights).
- S2 adv-reweight (wpow .5): c25 -0.00029, c26 -0.00050, c23 -0.00106 (w: -0.0005/-0.0006/-0.0022); 425 s
- S3 nfolds3: -0.00019/-0.00020; nfolds2: -0.00044/-0.00033; c23 nfolds1: -0.00136. nf5 (c26): -0.00002. negfrac1.0: +0.00004/+0.00021/-0.00014 (1002 s)
- LR member w0.2: +0.00048/+0.00065/+0.00086 (weighted +0.00037/+0.00066/+0.00084); LR fit 22-40 s (c25 390 s)
- adv AUC train-vs-eval: 0.9665 (cut fingerprint via item stats), 0.8975 w/o item stats (tB_*, history-length driven)
- seq gain by staleness (val23): <6h +0.0011, 6-24h +0.0016, 1-3d +0.0025, >3d +0.0015 -> not a stale-row failure
- eval sq27 recompute: corr(z,sq) 0.813 on eval (val 0.69-0.84); recon of shipped v8 corr 0.998 -> no eval seq bug visible
Probes: v2work/out/probe_lr.csv.zip, probe_sqonly.csv.zip
