# v7 lite options: AUC cost vs time saved (4-core)

Reference: full v7 blend 0.81164 (val25) / 0.82099 (val26). Single v7base s0 0.81081 / 0.82025. Val AUCs are from v7_q4/q5 (`v2work/v7_q4*.log`, `v7_q5.log`).

| Option | val25 | val26 | Δ vs full blend (25/26) | Time effect |
|---|---|---|---|---|
| (a) 1 base seed + slow | 0.81147 | 0.82079 | −0.00017 / −0.00020 | saves about 3.5 min (one fewer base fit) |
| (b) neg50 (+slow) | 0.81147 | 0.82075 | −0.00017 / −0.00024 | base fit about −50% |
| (c) 3 folds (+slow) | 0.81159 | 0.82081 | −0.00005 / −0.00018 | fit −28 to −41% |
| (d) top175 feats (+slow) | 0.81146 | 0.82069 | −0.00018 / −0.00030 | fit about −50% |
| (e) max_bin 63 (+slow) | 0.81154 | 0.82099 | −0.00010 / 0 | no measurable speedup |
| **cmb = a+b+c+e** blend | 0.81125 | 0.82059 | −0.0004 / −0.0004 | base 50 s, slow 67–95 s |
| **lite_nox2** = cmb without extra2 (base model) | 0.81053 | 0.82043 | −0.00011 / +0.00034 vs cmb base | saves 170–231 s of extra2 features |

Dropping extra2 is about AUC-neutral (mean +0.0001) and is the biggest single time saving.
**Recommended config: `lite_nox2`.** Expected blend about −0.0004 vs full v7 (about 0.8112 / 0.8206).

## Exact 4-core timing: `timed_run.py --config lite_nox2`

Run at 23:41–23:56 IST on 23 Oct 2026, pinned with `taskset -c 0-3` and all thread environment variables set to 4. Host: Intel Xeon, 8 vCPU, 15 GB. Log: `v2work/out/timings.jsonl`.

| Stage | Wall s | CPU s |
|---|---|---|
| load_prep_v3 | 18.5 | 37.4 |
| load_prep_v2 | 9.3 | 25.0 |
| feat_v3 | 161.1 | 273.4 |
| feat_v2 | **454.5** | 445.8 |
| feat_x | 62.0 | 154.2 |
| feat_lr | 75.4 | 98.1 |
| feat_tt | 46.9 | 136.2 |
| fit_predict_base | 79.5 | 275.0 |
| fit_predict_slow | 90.6 | 335.9 |
| blend + write + validate | 8.2 | 13.3 |
| **TOTAL** | **1006.0 (16.8 min)** | **1794.3** |

The submission passed validation (header, all rowids, values finite in [0,1]). sha256 `32208cffea42914228f611101c0ac8c0336f46032445d9c657471525d6cdfe4f`. It was **not uploaded**.

**Anomaly:** feat_v2's eval build took 365 s, against 19–23 s per fold and 118 s for the whole feat_v2 stage in the previous (aborted) run at 23:3x. CPU was roughly equal to wall, i.e. single-threaded stall; no v5 process was running, and load from other agents' work is the likely cause.
- With the previous run's 118 s feat_v2, the **clean estimate is about 670 s (11.2 min)**.
- Prep numbers are warm-page-cache; cold prep_v3 was 108 s earlier, which adds about 1.5 min on a fresh machine.
- Re-time feat_v2 alone to confirm.

Remaining hot spots: feat_v3 (161 s) and feat_v2 (118–455 s) are about 45–60% of the wall time, ahead of the model fits (170 s).
