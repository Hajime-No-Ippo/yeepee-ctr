# U1/U2 independent check (2026-11-02, Europe/Dublin)
- v21fit/blend_inputs/ does NOT exist on the box (only /workspace/v21fit_code_stage/v21fit/ scripts); rebuilt from v21fit/*.npy arrays (EN=erbrown/w10an/eb_new_rank.npy, E0=nr(merged.eb), A_all, D_dcn, inv_v8regen, inv_v5). Pipeline check: rebuilt S1 vs S1 zip max|diff|=5e-13.
- U1 csv sha c9ef4627...60f8, zip sha b8d923d2...61fb; U2 csv sha 9640e1fb...63e8, zip sha 0f1c2847...92a9. Eric's zip shas (7401b7a1..., 73363773...) NOT matched (zip bytes include timestamps; no copy of Eric's zips on box so CSV content cannot be compared; need CSV sha from the repo/Mac: `unzip -p U1.zip | shasum -a 256`).
- Scripts: build.py corr.py fit.py graph.py noise.py; out/ holds my rebuilt zips (not uploaded).
