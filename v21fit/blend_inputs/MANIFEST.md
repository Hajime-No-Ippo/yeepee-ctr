# v21fit_blend_inputs MANIFEST
Eval-set prediction rank arrays read by the v21fit blends (Q1, Q2, R1, S1, S2). Private dataset only. No labels, no competition data, no row ids.
- Shape of every array: (529390,) (R0.npy: 10x10). Row order: `erbrown/merged.parquet` rowid order == row order of submission_blend_newbest_v21_w10_20261031.csv.zip and of all v21fit probe csvs. sha256 of "\n".join(rowids) = `f4a805a3d6d3c60396dfa9ae915c2c20590fc043011a9621b6b5579f8e302742` (the blend script checks it).
- All arrays are nr(x)=(rankdata(x)-1)/(n-1), values in [0,1]. Machine-readable version: MANIFEST.json (same folder).
- Recipe and code: GitHub/HF `yeepee-ctr` -> `v21fit/BLEND_RECIPE.md`, `v21fit/blend_probes.py`.

| file | role | shape | dtype | bytes | sha256 | source |
|---|---|---|---|---|---|---|
| EN.npy | core | (529390,) | float64 | 4235248 | `6a606f7fa343f7e195ded3abd0b1b9b10e3985e3356854afe262e54ad17d25f6` | erbrown/w10an/eb_new_rank.npy (EB-new rank, ErBrown latest, public 0.828856; recovered as (W-0.1*O)/0.9 from submission_blend_newbest_v21_w10) |
| E0.npy | core | (529390,) | float64 | 4235248 | `e26e28f25563bd8f825c6acc7c23033c8d113a68034c3852dc2be82c0178131c` | rank-normalised M["eb"] of erbrown/merged.parquet = nr(eb) (ErBrown prior submission_best, public .824558); nr(x)=(rankdata(x)-1)/(n-1) |
| A_all.npy | core | (529390,) | float64 | 4235248 | `f707a4e7661f26b6d558347085760cbf0009af5b03bc51668e2cd7389845ad24` | v21fit/A_all.npy = v21seq A _all (8 seeds ctl6 listwise MLP + 3 history-model seeds), rank-normalised |
| D_dcn.npy | core | (529390,) | float32 | 2117688 | `946a077897efe51940f1a19cfbafe97575231244af5a59c25e2a8f2456c3cc59` | v21fit/D_dcn.npy = regularised DCN-v2, mean of 4 members (dcn6 s0, dcn6b s1, dcn7 s0, dcn4 s0, C27 eval), rank-normalised, built by mkD.py |
| v8regen.npy | core | (529390,) | float64 | 4235248 | `2652a969c6444fa19bf73edfec11bc67cc379afa58eefea3920ab1d31400730c` | v21fit/inv_v8regen.npy = nr(predicted_probability) of v12x/v8regen/submission_v8_regen.csv.zip aligned to merged rowid order |
| v5.npy | core | (529390,) | float64 | 4235248 | `f49f6390aaf89817bde6e10b18f16f2b287a6392708131d006a095843b77ab1b` | v21fit/inv_v5.npy = nr(predicted_probability) of v5/submission_v5.csv.zip aligned to merged rowid order |
| P_p33.npy | fit-only (binormal/Monte Carlo scripts addon*.py, reopt*.py, recal.py) | (529390,) | float64 | 4235248 | `6b28c316468b40d8997b510eb62368a44bfaa30a3ded2b06bde624f174467a60` | rank-normalised M["p33_rank"] (teammate p33) of erbrown/merged.parquet |
| S_ebdistill.npy | fit-only | (529390,) | float32 | 2117688 | `4c799478bd74022c6312993ef17cb20c98c20e4f150acfc16742809fa6fe0b53` | v21fit/S_ebdistill.npy = EBDISTILL student (trained on EB-new pseudo labels) |
| v21slateB.npy | fit-only | (529390,) | float64 | 4235248 | `e769b38b8d0f6e20d3d7d872ff625734fbb2c090600fb7bf68f7c48653b35f35` | v21fit/inv_v21slateB.npy = rank of v21diag submission_v21slate_B_TRANSDUCTIVE |
| v21listA.npy | fit-only | (529390,) | float64 | 4235248 | `28efb94f1edcca1e03490292151aeebc352cc192e196efbf0b80a9ec65fa74f2` | v21fit/inv_v21listA.npy = rank of v21diag submission_v21list_A_trainonly |
| R0.npy | fit-only | (10, 10) | float64 | 928 | `97ef7ba9c294f94e2b8352122e8e76ac245ff31710078de6d5531589383c90ad` | v21fit/R0.npy = 10x10 rank-correlation matrix (subsample 200k rows) of EN,E0,A,D,P,v5,v8r,S,slB,listA in this order (inputs of addon.py) |
