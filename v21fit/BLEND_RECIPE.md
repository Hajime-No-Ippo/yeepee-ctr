# v21fit blend recipe (Q1, Q2, R1, S1, S2)

Everything here is a **rank blend on top of ErBrown's EB-new**. No labels, no eval aggregates. Nothing is trained: it is a linear mix of 0-1 rank arrays followed by a re-rank.

## Rank transform
`nr(x) = (scipy.stats.rankdata(x) - 1) / (len(x) - 1)`  (average ranks for ties, result in [0,1]). Every input array is already `nr`-normalised. After mixing, the sum is re-ranked with the same `nr` and written as `predicted_probability` (12 decimals).

## Inputs (private HF dataset `Eric-Yyvuw/yeepee-ctr-cache`, folder `v21fit_blend_inputs/`; see its `MANIFEST.md` / `MANIFEST.json` for sha256, shape, dtype, source)
All arrays are 1-D, shape **(529390,)**, one value per eval row.

| symbol | file | dtype | what |
|---|---|---|---|
| EN | `EN.npy` | float64 | EB-new rank (ErBrown latest), public **0.828856** |
| E0 | `E0.npy` | float64 | EB-old rank (ErBrown previous, `submission_best`), public .824558; rho(EN,E0)=.985 |
| A | `A_all.npy` | float64 | v21seq A `_all` (8 seeds listwise MLP + 3 history-model seeds), ~.819-.8197 public; rho(EN)=.956 |
| D | `D_dcn.npy` | **float32** | regularised DCN-v2, 4 members; rho(EN)=.950, rho(A)=.9935 |
| V8 | `v8regen.npy` | float64 | older v8 regen line (~.8145 public est.) |
| V5 | `v5.npy` | float64 | older v5 line (~.79 public est.; no real public score) |

Fit-only extras (not needed for the five probes, only for the binormal / Monte-Carlo scripts `addon*.py reopt*.py recal.py inv.py`): `P_p33.npy`, `S_ebdistill.npy`, `v21slateB.npy`, `v21listA.npy`, `R0.npy` (10x10 rank-corr matrix, order EN,E0,A,D,P,v5,v8r,S,slB,listA).

**Row order**: `merged.parquet` rowid order (`erbrown/merged.parquet`), which is the row order of `submission_blend_newbest_v21_w10_20261031.csv.zip` and of every v21fit probe csv. Row ids themselves are not stored in the cache dataset. `blend_probes.py` reads them from any of your existing probe zips (`--rowid-from`) and checks `sha256("\n".join(rowids)) == f4a805a3d6d3c60396dfa9ae915c2c20590fc043011a9621b6b5579f8e302742`.

## Formulas (weights are on the rank arrays; negative = subtracted)
General form: `X = EN - c*E0 + g*(1-q)*A + g*q*D + w8*V8 + w5*V5`, then `nr(X)`.

| probe (file prefix `probe_v21fit_...`) | c (E0) | g (ours total) | q (D share of ours) | w8 (V8) | w5 (V5) | public |
|---|---|---|---|---|---|---|
| EN itself | - | - | - | - | - | **0.828856** |
| `Q1_EBnew_m136E0_p098_A50_DCN50` | .136 | .098 | .5 | 0 | 0 | pending (fit predicts ~.82895) |
| `Q2_EBnew_m136E0_p098DCN` | .136 | .098 | 1.0 | 0 | 0 | **0.829016** (+.000160) |
| `R1_Q1_m030v8regen_m015v5` | .136 | .098 | .5 | -.03 | -.015 | **0.829058** (+.000202, best so far) |
| `S1_R1step_c109_g123_q67_v8m052_v5m033` | 0.10902810947726392 | 0.12251977564719992 | .665 | -0.052490334215038625 | -0.033138516122339604 | pending |
| `S2_R1step_c097_g129_q75_v8m079_v5m015` | 0.09693596102582225 | 0.1285964186161041 | .75 | -0.078587083614776 | -0.015 | pending |

Explicitly: Q1 = EN - .136 E0 + .098(.5 A + .5 D); Q2 = EN - .136 E0 + .098 D; R1 = Q1 - .03 V8 - .015 V5.
S1/S2 are 1/3 and 1/2 steps from R1 toward the optimum of the posterior fit after R1 and Q2 public scores (`recal.py`, `reopt2-4.py`, `mkS.py`).
(Older probes P1 = EN - .136 E0 + .098 A, P2, P3 use EBDISTILL; see STATUS.md. Not rebuilt by `blend_probes.py`.)

## How to run (numpy + scipy only, ~10 s)
```
# 1. download inputs (about 38 MB; core 6 arrays are about 21 MB)
hf download Eric-Yyvuw/yeepee-ctr-cache --repo-type dataset --include 'v21fit_blend_inputs/*' --local-dir ~/Downloads/huawei/yeepee_cache_dl
# 2. rebuild probes (any v21fit probe zip supplies the rowid order)
python v21fit/blend_probes.py --inputs ~/Downloads/huawei/yeepee_cache_dl/v21fit_blend_inputs \
   --rowid-from ~/Downloads/huawei/probe_v21fit_Q2_EBnew_m136E0_p098DCN.csv.zip --out ./v21fit_rebuilt [--only Q1,R1]
```
Check: the **unzipped CSV** sha256 must match (the zip file hash differs because of zip timestamps; the original zip hashes are in `SHA256SUMS_v21fit_probe_zips.txt`):
```
Q1 d1fc7760860bba6bee72a9c0a144bc585dd16063e70985fc55d401fc929ac805
Q2 85ab3e52a8e5cc8209bfaa16e06be63ffbf51a190d49473d8134ccc86b0dd5c3
R1 7f20cb95e1eab7a171c13ec6c468ceb64eeeac50fd48fd92f6f7de15b02532bd
S1 24279c3d946a3eab0c85b1707fff842f95cacb2b245d84c8bb754c5ba51e25ed
S2 c07841bc942902e1a73c0b7800ee88c99da6482cc315c467fbb00c3fcdb93c00
```
(`unzip -p X.csv.zip | shasum -a 256`). On the box, `blend_probes.py` reproduced all five byte-for-byte against the original probe CSVs.

## Notes
- Keep D as float32 as stored; the original run used it that way and the operation order in `blend_probes.py` mirrors mkQ/mkR/mkS, otherwise a handful of near-tied rows can move one rank step (~2e-6, harmless).
- Private inputs stay in the private cache dataset only (access needs your HF login). This repo never holds predictions.
- Fit story and calibration (binormal AUC model, posterior from public scores R1/Q2): `STATUS.md`, `recal.py`, `reopt2.py`-`reopt4.py`, `addon*.py`.

## Update 2026-11-02: inputs are also in this repo (one-off exception, approved by Eric)
The six core arrays (`EN.npy E0.npy A_all.npy D_dcn.npy v8regen.npy v5.npy`, D_dcn stays float32) plus `MANIFEST.md` / `MANIFEST.json` are committed under **`v21fit/blend_inputs/`** (about 22 MB total, each file under 5 MB). This is a one-off exception to the "no predictions in code repos" rule and covers only these files; the scheduled sync still excludes `.npy` elsewhere. The fit-only extras stay only in the private cache dataset.
Default `--inputs` is therefore `v21fit/blend_inputs` (no download needed):
```
python v21fit/blend_probes.py --inputs v21fit/blend_inputs --rowid-from <any v21fit probe zip> --out ./v21fit_rebuilt
```
Verify files against `v21fit/blend_inputs/MANIFEST.json` (sha256, shape, dtype). The private cache folder `v21fit_blend_inputs/` in `Eric-Yyvuw/yeepee-ctr-cache` holds the same bytes.
