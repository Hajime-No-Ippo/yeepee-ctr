# Teammate AI's reply about p33 (pasted by Eric, 2026-10-29 15:06 Dublin)

- p33 = experiment script #33, p33_final_submit.py, generates the final submission. Naming: p<n>_<content>.py (p3_w2v_fusion.py = W2V multi-config fusion; p22_mlp.py = MLP).
- Final submission = 0.75*rank(teammate v8) + 0.25*rank(our model); our model = 0.7*rank(LightGBM artA) + 0.3*rank(MLP).
- Pipeline: 1) src/full/lgbm_full.py streams 7.3GB CSV into event table + row table; 2) features: 230 dims from history x item (p3_w2v_fusion.py, p9_coords.py); 3) models LightGBM/XGBoost/CatBoost/MLP (p30_ctrparams.py, p21_xgb.py, p22_mlp.py); 4) rank average + blend with teammate (p33_final_submit.py); 5) src/tools/check_submission.py.
- Key feature: item x item co-occurrence matrix over all history co-occurrences in 6.87M rows -> PPMI -> SVD to 12-16 dims -> user vector = recency-weighted mean of history item vectors -> similarity = user . candidate.
- Rank probe: k=16 best (0.7551), k=256 collapses to 0.672 -> claims data is low-rank (synthetic, 16-dim latent factors).
- Roadmap AUCs: baseline ~0.50; basic CTR features ~0.68; time discipline (train first 5 days, validate day 6; stats only pre-event); PPMI+SVD affinity ~0.78; multi-config + coords (k=12/16 spaces, item/user coords, position decay) ~0.81; ensemble LGB+XGB+Cat+MLP ~0.814; blend with teammate 0.8153.
- Pitfalls: IDs as numeric (AUC 0.489); time leakage; chasing local gains (local +0.001 -> public +0.00015).
- Docs in D:\onner12\docs: 改进记录_初学者版.md, FINAL_HANDOVER.md (6 core findings), 解题思路与特征重要度.md, EXPERIMENT_LEDGER.md (40+ experiments incl. rejected), GAP_AUDIT.md, D16_LOW_RANK.md.
- Scripts: src/full/lgbm_full.py, experiments/p3_rank_probe.py, p3_w2v_fusion.py, p9_coords.py, p30_ctrparams.py, p22_mlp.py, p21_xgb.py, p33_final_submit.py (6.9KB).
- Data: data_full\lightgcn\data.npz; feature blocks 12 configs x 3 time points data_full\p3\<cfg>\{W45,W6,We}.npy; base 38 features data_full\w2v_rolling\features\X45.npy/X6.npy; final model data_full\p3\p33_lgb_artA_day46.txt (trained day 4-6?).
- They have a copy of our repo at D:\onner12\teammate\yeepee-ctr-main; they say our local 0.82204 vs theirs 0.81374. Files they cite: v8b/sol/solution_v8.py, v8/seq.py (DIN), v3/features.py (170 cols), v3/prep.py, v9d/report_v9d.md, research/*.md.
