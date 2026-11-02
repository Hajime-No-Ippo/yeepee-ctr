# BLEND_RECIPE.md (created 01:25 Nov 2; no earlier file existed on the box; append only)
All files: rank(nr) of EN - c*E0 + g*(1-q)*A + g*q*D + v8*v8regen + v5*v5 ; arrays in merged.parquet order
(EN, E0 from lib.py; A=A_all.npy; D=D_dcn.npy (4-snapshot DCN bag); v8regen=inv_v8regen.npy; v5=inv_v5.npy). Written by mkQ.py/mkR.py/mkS.py/mkT.py.
name | c(E0) | g(total ours) | q(DCN share) | v8regen | v5 | public / predicted gain vs EN or R1
P1 | .136 | .098 | 0 | 0 | 0 | not scored
Q1 | .136 | .098 | .5 | 0 | 0 | not scored (pred +.000096, AUC .828952)
Q2 | .136 | .098 | 1 | 0 | 0 | .829016
R1 | .136 | .098 | .5 | -.03 | -.015 | .829058
S1 | .109 | .123 | .665 | -.052 | -.033 | pred +.000128 vs R1
S2 | .097 | .129 | .75 | -.079 | -.015 | pred +.000114 vs R1
T1 (a1) | .081 | .148 | .834 | -.075 | -.052 | pred +.000224 vs R1
T2 (a2) | .054 | .172 | 1.0 | -.098 | -.070 | pred +.000283 vs R1
T3 (b)  | .077 | .144 | .875 | -.103 | -.015 | pred +.000157 vs R1
T4 (c)  | .136 | .098 | .75 | -.03 | -.015 | pred +.000027 vs R1
T1 (a1) low confidence after S1=.829101: pred +.00003..+.00004 vs S1.
Refit 01:57 (5 scores): S1 (c .109 g .123 q .665 v8 -.052 v5 -.033) remains best; no point on R1-S1-T1 line beats it by >.00001.
