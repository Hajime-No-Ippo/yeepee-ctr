# cb_blend: v21cat + v21graph members (Grok Bot executor, 2026-11-02 ~03:45 Europe/Dublin)
Members frozen in cb_blend/members/ (cp -p at 03:01; graph_simple = 02:57 file, sha 7a1a7d4a.. / 45163c96.., = v21graph STATUS 03:15 interim: C23 u1b n=4 seeds, C25 u1 n=5).
Alignment: v21cat/cache/val{c}_rowid.npy, v21graph/out/tab_val{c}.npz rowid and v21seq/seqfeat/A_val{c} rowid all == v12/feat_v3/val{c}.parquet rowid (23/25/26, n 501550/439640/351380).
| member | C23 | C25 | C26 | rho_M (Spearman) C23/C25/C26 |
|---|---|---|---|---|
| cat (reported .7965/.8164/.8249) | .79654 | .81638 | .82493 | .9760/.9803/.9819 |
| graph (reported .765/.796) | .76508 | .79646 | - | .8104/.8715/- |
Held-out (weight picked on other fold), rank space, w = weight of new member in (1-w)*eq4 + w*new:
- graph: fit C25 -> w=.15 -> C23 .79951 (+40.1e-5 vs eq4 .79911); fit C23 -> w=.10 -> C25 .82012 (+60.1e-5 vs eq4 .81952).
- cat: fit C25 -> w=.10 -> C23 .79920 (+9.0); fit C23 -> w=.15 -> C25 .81953 (+2.0).
Recommended: eq4+graph .1 = z,M,big,D .225 each + graph .10: C23 .79960 (+49 vs eq4, user-subset +49+-12) / C25 .82012 (+60, +60+-17); rho_M .9954/.9972.
eq4+cat.05+graph.1: C23 .79964 / C25 .82013 (cat adds +4/+1, noise). Full search: search_nov2.log, make_report_nov2.out, final_eval_nov2.log, blend_new.log.
