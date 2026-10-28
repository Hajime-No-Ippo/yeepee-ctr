# v15cat — CatBoost with categorical sets: **NO-GO** (verdict 07:40 Dublin/UTC)
Rule: NO-GO if run A or B < dwbig alone on C25 (.81658), or adds < +.0002 C25 as blend member to v12 (.81723), or hurts C23.
Setup: catboost 1.2.10 CPU, depth 8, lr .05, l2 10, max_ctr_complexity=1, border 128, od_wait 100, neg-frac .5, train f1..f4 (cut 25), early stopping ON val25 (so numbers are optimistic), 8 threads.
Cats (21): raw item_id/user_id, impression-SVD clusters item 64/256/1024 & user 64/256/1024, last-history-item cluster 64/256, rec/hlen/stale bins, time_bucket, day_type, in_hist, 5 crosses.
| run | features | best_iter | C25 AUC | fit s | blend into v12 (w=.5/1/2) |
| A  | 21 cats + top-100 numeric | 35 | .80884 | 25 | .81704/.81672/.81599 |
| B  | 21 cats + all 435 numeric | 34 | .80798 | 44 | .81704/.81670/.81593 |
| A no raw ids (diag) | 19 cats + top100 | 847 | .81450 | 153 | .81726/.81723/.81707 |
| A no cats (diag) | top100 numeric | 552 | .81500 | 44 | .81722/.81716/.81701 |
v12 = .81723; dwbig alone .81658. Both A and B are below dwbig by ~.008 (val AUC peaks at ~35 iterations then falls: raw user/item-id cats overfit/shift under time split); even diagnostic variants stay below dwbig and add <= +.00003 as members. C23 not run (rule already failed on C25). CatBoost work stopped.
