# Graph model (LightGCN "u1") — Yeepee, Huawei TechArena CTR
Built by Grok Bot in /workspace/huawei/v21graph on the shared Linux box (not on the Mac/repo). Packaged by Data Wrangler.
Leakage: graph = TRAIN-ONLY click graph at each cutoff (train events ts < J20 + k*DAY; k=3 C23, 5 C25, 6 C26, 7 eval). No eval-row labels/aggregates -> NOT transductive.

## Model
LightGCN d64 L3 lr1e-2 l2 3e-3, BPR + slate-softmax loss; 350 steps (k5/k6/k7), 800 steps for C23 (u1b, k3). Steps were picked with val monitors (mild selection).
Score G = z(s_uv)+z(s_h)+z(cos_mean), each averaged over 7 seeds {0,1,2,3,4,5,9}, raw (not demeaned). Code: code/gcn.py (train), gfeat.py (features), final.py / fin2.py (seed average + save), mkG2.py (blend). Full notes: code/STATUS.md.

## preds/ (all aligned row-for-row within each cutoff)
- graph_c23/c25/c26.npy + y_c23/c25/c26.npy (labels) + M_ctl6A_rank_c*.npy (our MLP ref "M", 8-seed rank). Fold row order = v21seq val-table order.
  Check: AUC(y_c23,graph_c23)=.76818, C25 .79514, C26 .80560.
- graph_eval.npy = feat_v3/eval.parquet rowid order; graph_eval_v21seqorder.npy + graph_eval_rowid.npy = v21seq order.
- graph_eval_merged_rank.npy and EN_eval_rank_merged.npy = merged/submission order (529,390). rho(graph, EN) on eval = .905.
  EN (ErBrown's best file) exists only on eval rows, so rho vs EN can only be measured on eval; no fold preds for EN.

## submission/
probe_v21graph_G2_S1_p50graph7s.csv.zip = nr(S1 + 0.05*nr(graph)) -> public 0.829113 (best). S1 .829101; G1 (-0.045) .828955.
Public-score fit implies graph eval AUC ~.799 (90% ~.796-.802), best w ~.04-.05.

Embeddings (~2 GB, emb_*.npz) stay on the box in v21graph/out/.
