# v21graph3 STATUS (feature-aware deep graph on G8 L8 base). Started 2026-11-02 ~15:00 Dublin
BAR (raised by Eric, 15:00): candidate passes locally ONLY if adds >= +5e-4 on BOTH C23 and C25 over base(.4z+.6M)+G8 (tuned w) AND eval rank rho vs EN < .90.
Then delivery bar unchanged: binormal fit (v21fit2 fitg3) predicts >= +5e-5 over G3 (.829117), P(beat G3)>.75 -> ONE file, probe_v21graph3_<recipe>.csv.zip -> Mac ~/Downloads/huawei + shasum. Eric uploads. Otherwise no file.
Rules: train-only, cores 4-7 (taskset), RSS<6-7GB, no uploads/pushes, copy don't modify v21graph/v21graph2.
## Notes
- Data schema: NO item side attributes (items/users opaque). Fields: user,item,timestamp,time_bucket,day_type,history(<=20 clicks + ts). Slate position carries no signal (click pos uniform). So 'feature-aware' = context (time bucket/day type/hour) + pointwise head w/ biases + no-click events.
- slates.npz (v21graph2/out): 686824 events, 254801 no-click (pos -1), 432k with click. NU=53003 NI=50000. Graph k cutoffs: C23=k3 C25=k5 C26=k6 eval=k7. L8 run ~7 min/seed (2 thr).
- 15:20 gcn3.py = gcn2 + pointwise BCE head (--pw: logit=<U,V>+b_i+b_u on ALL slate candidates incl. 254k no-click events) + --ctx (hour x daytype embedding added to user vec). ev3.py scores/gains on top of base+G8. Check: CTR vs hour/day flat (.624-.632), no-click rate depends on hist length only (.43 empty -> .645 full).
- p1 (pw+slate+bpr, 1 seed k5): standalone C25 s3b .7964; on top of base+G8(w.2, +139.6e-5): best +16e-5 (s3b+bi+bu v.05) vs bar +500 => FAIL. p2 (ctx): embeddings unusable without ctx (.609) -> dropped (ctx has no signal). Biases alone ~.52 AUC. Running q1 (pw only), q2 (pw+slate, no bpr).
