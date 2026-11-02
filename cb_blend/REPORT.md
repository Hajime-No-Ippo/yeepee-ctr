# cb_blend REPORT (Code Buddy, 2026-11-02 02:25 Europe/Dublin)
Out-of-fold stack / weight search over 'our own' fold predictions, variant A only, global rank/N only, no eval file built, nothing uploaded. Work folder: /workspace/huawei/cb_blend (everything else read-only). CPU only, cores 0-3, 4 threads, RSS < 2 GB.
Files: blend_oof.py (members config + search), make_report.py, final_eval.py, oof_c{23,25,26}.npz (aligned raw member scores + y + rowid), search_out.pkl, final_eval.json, best_weights.json, search.log.

## TL;DR
* Best regularised 'ours' combination: **equal weights z .25 / M .25 / big .25 / D .25** (rank/N). Held-out-style numbers: C23 0.79911 (+8.8e-5 vs .4z+.6M), C25 0.81952 (+30.2e-5), rho vs M .9976 / .9986. Cross-fold fitted + ridge variants of the same set gave +8.4 (C23) / +29.8 (C25) held-out (LR-L2), i.e. same answer; unfitted equal weights are as good as any fitted weights.
* Caveats: (1) D at C25 is the v21fit D3 bag whose snapshots/configs were chosen on C25, so C25 gain (+30) is optimistic; at C23 no matching DCN bag exists, I used a PROXY (v21dcn r1 configs at peak epoch, AUC .79793), so C23 is the cleaner test (+9, ~1.8 SE by user-subset spread). (2) Gains over base are small (1e-4 to 3e-4); fold-to-fold noise is of the same order. (3) C26 has only M/big (no z, list, D, ref): M/big 50/50 .82798 = A_all, +.00001..+.00002 over M alone.
* **v20 ref484 does not help reliably**: rho vs M .980 (C23) / .985 (C25) (z: .979/.992, D: .992/.992), standalone .79707/.81778 (3-seed avg) is .0016/.0013 below M. Adding to eq4 (eq5, .2 each): C23 -3.2e-5, C25 +3.1e-5 (user-subset sd ~5-10e-5); without D (z,M,big -> +ref): C23 -10.4e-5, C25 +4.9e-5. Cross-fold fits: with ref the C25-fit gives it weight .24-.27 (mostly replacing z, weight 0) but the C23 held-out delta goes +8.4->+6.8 (LR) / +5.4->+7.9 (shrunk) / -0.3->+3.2 (plain): sign flips, so no robust gain. Keep out (or at most .1-.2 weight if Eric wants to hedge; in-sample best 'sh5' below uses it).
* No alignment problems: every member's standalone AUC reproduces the reported value to <=5e-6 (see section 1). Only caveat: D at C23 (no reference, proxy), and C26 lacks z/list/D/ref.

## 1. Alignment (rowid + n + standalone AUC check, tol 1.6e-4)
Labels/rowid = v12/feat_v3/val{cut}.parquet (identical rowid list to v9/feat_v3). ctl6/big rows verified rowid-equal to v21seq/seqfeat/A_val{cut}.npz (23/25/26 all True; n=501550/439640/351380). z, list1, D, ref rows verified through reproducing reported AUCs (z .79679/.81829, M .79862/.81904/.82797, big .79851/.81916/.82791, A_all .79864/.81910/.82798, list1 .79799/.81857, D3 C25 .8189, ref484 3-seed-avg .79707/.81778, individual seeds .79665/.79657/.79670 and .81748/.81748/.81752 = reported .79664/.81749 mean-of-seeds).

| member | C23 AUC | C25 AUC | C26 AUC | rho vs M C23 / C25 / C26 |
|---|---|---|---|---|
| z (v19 stack / dwbig) | .79679 | .81829 | - | .9789 / .9918 / - |
| M = ctl6 8 seeds | .79862 | .81904 | .82797 | 1 |
| big = v21seq_big (3/3/2 seeds) | .79851 | .81916 | .82791 | .9989 / .9994 / .9993 |
| A = A_all (ctl6 8 + big) | .79864 | .81910 | .82798 | .9999 / 1.0000 / 1.0000 |
| list1 = v21list seed0 | .79799 | .81857 | - | .9966 / .9976 / - |
| D = DCN-v2 bag (C23 = proxy) | .79793 | .81890 | - | .9917 / .9922 / - |
| ref = v20 ref484 (3 seeds) | .79707 | .81778 | - | .9803 / .9848 / - |

## 2. Baselines (rank space; raw logit space in brackets, Grok Bot's numbers)
.4z+.6M: C23 .79902 [.79903] (Grok .79902/3), C25 .81921 [.81921]. .4z+.6A_all: C23 .79904 [.79905], C25 .81925 [.81925]. .3z+.3M+.4D: C25 .81949 [.81950] (Grok .81949), C23 .79900 with proxy D (-2.1e-5: no gain at C23 in that mix). All reproduced.

## 3. Held-out candidates (weights fitted on ONE fold, scored on the OTHER). delta in 1e-5 AUC vs .4z+.6M (rank space) on the same fold

| member set | procedure | weights (order = set) | fit C25 -> held-out C23 | delta | fit C23 -> held-out C25 | delta | mean held-out delta (1e-5) |
|---|---|---|---|---|---|---|---|
| z,M,big,D,ref | shrunk 50% to equal | C25-fit 0.10,0.10,0.31,0.27,0.22 / C23-fit 0.18,0.26,0.22,0.18,0.17 | 0.79910 | +7.9 | 0.81955 | +33.7 | +20.8 |
| z,M,big,D,ref | LR-L2 | C25-fit 0.19,0.20,0.20,0.21,0.20 / C23-fit 0.19,0.21,0.21,0.20,0.19 | 0.79909 | +6.8 | 0.81956 | +34.3 | +20.6 |
| z,M,big,D | LR-L2 | C25-fit 0.25,0.24,0.25,0.26 / C23-fit 0.26,0.25,0.25,0.24 | 0.79911 | +8.4 | 0.81951 | +29.8 | +19.1 |
| z,M,big,D,ref | plain fit | C25-fit 0.00,0.00,0.43,0.33,0.24 / C23-fit 0.17,0.32,0.23,0.15,0.13 | 0.79906 | +3.2 | 0.81953 | +32.1 | +17.7 |
| z,M,big,D | shrunk 50% to equal | C25-fit 0.22,0.12,0.34,0.32 / C23-fit 0.26,0.29,0.25,0.20 | 0.79908 | +5.4 | 0.81949 | +27.4 | +16.4 |
| z,M,D,ref | plain fit | C25-fit 0.00,0.39,0.35,0.25 / C23-fit 0.13,0.55,0.16,0.16 | 0.79905 | +2.5 | 0.81951 | +29.6 | +16.0 |
| z,M,D,ref | shrunk 50% to equal | C25-fit 0.12,0.32,0.30,0.25 / C23-fit 0.19,0.40,0.20,0.21 | 0.79904 | +1.4 | 0.81951 | +29.9 | +15.7 |
| z,M,big,D,ref | shrunk 50% to base | C25-fit 0.20,0.30,0.21,0.17,0.12 / C23-fit 0.28,0.46,0.12,0.08,0.07 | 0.79913 | +11.1 | 0.81939 | +18.1 | +14.6 |
| z,M,D | LR-L2 | C25-fit 0.32,0.34,0.34 / C23-fit 0.33,0.35,0.33 | 0.79903 | +0.8 | 0.81948 | +26.7 | +13.7 |
| z,M,D,ref | LR-L2 | C25-fit 0.21,0.28,0.29,0.23 / C23-fit 0.20,0.32,0.27,0.21 | 0.79898 | -4.2 | 0.81952 | +31.1 | +13.4 |
| z,M,D | shrunk 50% to equal | C25-fit 0.27,0.36,0.37 / C23-fit 0.31,0.44,0.25 | 0.79904 | +1.7 | 0.81946 | +24.8 | +13.2 |
| z,M,D,ref | shrunk 50% to base | C25-fit 0.20,0.50,0.18,0.13 / C23-fit 0.27,0.57,0.08,0.08 | 0.79912 | +9.5 | 0.81938 | +16.9 | +13.2 |
| z,M,big,D | plain fit | C25-fit 0.18,0.00,0.43,0.38 / C23-fit 0.28,0.32,0.24,0.16 | 0.79902 | -0.3 | 0.81945 | +23.9 | +11.8 |
| z,M,big,D | shrunk 50% to base | C25-fit 0.29,0.30,0.22,0.19 / C23-fit 0.34,0.46,0.12,0.08 | 0.79912 | +9.7 | 0.81935 | +13.5 | +11.6 |
| z,M,big,ref | shrunk 50% to equal | C25-fit 0.12,0.12,0.47,0.28 / C23-fit 0.22,0.24,0.34,0.20 | 0.79906 | +3.6 | 0.81940 | +18.4 | +11.0 |
| z,M,big,ref | plain fit | C25-fit 0.00,0.00,0.69,0.31 / C23-fit 0.18,0.23,0.44,0.15 | 0.79903 | +0.8 | 0.81942 | +20.2 | +10.5 |
| z,M,D | plain fit | C25-fit 0.22,0.38,0.40 / C23-fit 0.28,0.55,0.17 | 0.79902 | -0.1 | 0.81941 | +20.0 | +9.9 |
| z,M,D | shrunk 50% to base | C25-fit 0.31,0.49,0.20 / C23-fit 0.34,0.57,0.09 | 0.79910 | +7.9 | 0.81933 | +11.6 | +9.8 |
| z,M,big,ref | shrunk 50% to base | C25-fit 0.20,0.30,0.35,0.15 / C23-fit 0.29,0.41,0.22,0.08 | 0.79909 | +7.1 | 0.81933 | +11.5 | +9.3 |
| z,M,big,ref | LR-L2 | C25-fit 0.23,0.26,0.26,0.24 / C23-fit 0.23,0.27,0.27,0.23 | 0.79900 | -2.2 | 0.81937 | +15.9 | +6.8 |
| z,M,big | LR-L2 | C25-fit 0.33,0.33,0.34 / C23-fit 0.33,0.33,0.33 | 0.79908 | +5.2 | 0.81929 | +7.9 | +6.5 |
| z,M,big | shrunk 50% to equal | C25-fit 0.31,0.17,0.52 / C23-fit 0.32,0.38,0.29 | 0.79906 | +3.8 | 0.81929 | +7.4 | +5.6 |
| z,M,big | shrunk 50% to base | C25-fit 0.34,0.30,0.36 / C23-fit 0.36,0.52,0.13 | 0.79907 | +4.8 | 0.81925 | +4.0 | +4.4 |
| z,M,big | plain fit | C25-fit 0.28,0.00,0.72 / C23-fit 0.31,0.43,0.25 | 0.79902 | +0.1 | 0.81928 | +6.8 | +3.5 |
| z,M | plain fit | C25-fit 0.30,0.70 / C23-fit 0.30,0.70 | 0.79906 | +4.0 | 0.81923 | +1.6 | +2.8 |
| z,M | shrunk 50% to base | C25-fit 0.35,0.65 / C23-fit 0.35,0.65 | 0.79905 | +3.1 | 0.81923 | +1.2 | +2.2 |
| z,M | shrunk 50% to equal | C25-fit 0.40,0.60 / C23-fit 0.40,0.60 | 0.79902 | -0.1 | 0.81921 | -0.1 | -0.1 |
| z,M | LR-L2 | C25-fit 0.49,0.51 / C23-fit 0.44,0.56 | 0.79891 | -11.2 | 0.81919 | -1.9 | -6.5 |

## Fixed (unfitted) references, rank space

| candidate | C23 | delta | C25 | delta |
|---|---|---|---|---|
| base .4z+.6M ({'z': 0.4, 'M': 0.6}) | 0.79902 | +0.0 | 0.81921 | +0.0 |
| .4z+.6A_all ({'z': 0.4, 'A': 0.6}) | 0.79904 | +1.3 | 0.81925 | +3.4 |
| .3z+.3M+.4D ({'z': 0.3, 'M': 0.3, 'D': 0.4}) | 0.79900 | -2.1 | 0.81949 | +28.0 |
| .3z+.3A+.4D ({'z': 0.3, 'A': 0.3, 'D': 0.4}) | 0.79901 | -1.6 | 0.81951 | +29.7 |

Raw (logit-space) reproduction of Grok Bot numbers: .4z+.6M raw C23 0.79903, .4z+.6A raw C23 0.79905, .4z+.6M raw C25 0.81921, .4z+.6A raw C25 0.81925, .3z+.3M+.4D raw C25 0.81950

## 4. Shortlisted fixed-weight candidates (no fitting; both folds in-sample for the FIXED weights, but weights are not tuned except 'sh5')
delta in 1e-5 vs .4z+.6M in brackets. Last column = mean±sd (1e-5) of the delta across 8 user-hash subsets of the fold (C23 / C25), an uncertainty indicator.

| candidate | weights | C23 | C25 | C26 | rho vs M (C23 / C25) | user-subset delta |
|---|---|---|---|---|---|---|
| base .4z+.6M | z 0.4, M 0.6 | 0.79902 (+0.0) | 0.81921 (+0.0) | n/a | 0.9966 / 0.9987 | +0±0 / +0±0 |
| eq4 (z,M,big,D) .25 each | z 0.25, M 0.25, big 0.25, D 0.25 | 0.79911 (+8.8) | 0.81952 (+30.2) | n/a | 0.9976 / 0.9986 | +9±14 / +30±12 |
| eq5 (z,M,big,D,ref) .2 each | z 0.2, M 0.2, big 0.2, D 0.2, ref 0.2 | 0.79908 (+5.6) | 0.81955 (+33.3) | n/a | 0.9958 / 0.9974 | +6±8 / +33±11 |
| eq4b (z,M,big,ref) .25 each | z 0.25, M 0.25, big 0.25, ref 0.25 | 0.79897 (-5.3) | 0.81934 (+12.8) | n/a | 0.9950 / 0.9972 | -5±9 / +13±6 |
| eq3 (z,M,big) 1/3 | z 0.333, M 0.333, big 0.333 | 0.79908 (+5.2) | 0.81929 (+7.9) | n/a | 0.9975 / 0.9990 | +6±9 / +8±3 |
| eq4c (z,M,D,ref) .25 each | z 0.25, M 0.25, D 0.25, ref 0.25 | 0.79890 (-12.8) | 0.81945 (+23.4) | n/a | 0.9936 / 0.9961 | -13±7 / +24±14 |
| eq3d (z,A=M+big,D) .3/.3/.4 | z 0.3, M 0.15, big 0.15, D 0.4 | 0.79901 (-1.4) | 0.81952 (+31.0) | n/a | 0.9958 / 0.9974 | -1±13 / +31±19 |
| sh5 (50% mean-of-folds fit + 50% equal) | z 0.1, M 0.135, big 0.306, D 0.225, ref 0.233 | 0.79911 (+9.0) | 0.81962 (+40.6) | n/a | 0.9967 / 0.9976 | +9±14 / +40±13 |

'sh5' = 50% mean-of-folds fit over z,M,big,D,ref + 50% equal: best in-sample (+9 / +41) but is in-sample on both folds and uses ref/proxy D; not recommended.
Cross-fold LR-L2 stacking (ridge on ranks) collapses to ~equal weights; it is as good as the best fitted variants, worse for the 2-member z,M case (-6.5e-5 mean: it pushes z weight to ~.5).

## 5. Mean-of-folds objective
Weights maximising mean(C23,C25) AUC (in-sample, no held-out fold): z,M: .31/.69; z,M,big: .31/.24/.46; z,M,big,D: .25/.06/.41/.28; +ref: 0/.07/.41/.25/.27. M and big are rho .9994 so their split is not identifiable; z falls to 0 only when ref (a GBDT) is added.

## 6. How to add members
Edit MEMBERS in blend_oof.py (one line per member: {cut: loader}), e.g.
  'cat': {c:(lambda c=c: mean_logit(f'{H}/v21cat/ens/cat_A_c{c}_s*.npy')[0]) for c in (23,25)},
add the REPORTED AUC for the alignment check, add its set to `sets` in blend_oof.py main (or pass --members z,M,big,D,cat), then
  taskset -c 0-3 /workspace/huawei/v20/.venv/bin/python blend_oof.py --rebuild && python make_report.py && python final_eval.py
(`--rebuild` only needed to refresh cached oof_c*.npz). Search takes ~16 min for 7 sets (5-member Nelder-Mead on 4 threads).
