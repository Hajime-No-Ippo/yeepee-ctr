"""Huawei TechArena 2026 Phase 1 - submission_v3 pipeline (driver).
NO aggregates over evaluation rows: all global/item-level stats (item popularity, co-occurrence, transitions, SVD item
factors, item-side centroids, clusters) use training data only; each eval row's own history is used only for that
user's own features (user vectors folded in onto train-only item factors). Within-event (same 10 candidates) z-scores kept.
Features: v3 features (v2work/v3feat_ne.py, copy of v3/features.py) + v2 features (feats.py) + similarity extras
(extra.py: contrastive centroid cosines k32/64/128, recency-weighted, item-side, user-cluster CTR) + extra2.py (ridge taste
vectors, user-state). Time/gap features dropped ('notime').
Model: LightGBM binary, lr 0.05, 63 leaves, 400 rounds, 3 seeds, plain mean (no per-event normalisation).
Training: first event per user in [Ck, Ck+1d), Ck = Jan21..Jan26 (stats frozen at Ck). Eval stats frozen at Jan27.
Local validation (cutoff Jan25, train Jan21..24 folds): AUC 0.7775 (warm 0.7793, history-only 0.7230, empty 0.5602);
v2 0.7620. Sources: solution_v3_src.zip (v2work + v3 code)."""
import subprocess, sys
PY = sys.executable
V3 = '/workspace/huawei/v3'; V2 = '/workspace/huawei/v2work'
def run(cmd, cwd, env=''): print('>>', cmd); subprocess.run(f'{env} {cmd}', shell=True, check=True, cwd=cwd)
if __name__ == '__main__':
    run(f'{PY} prep.py /workspace/huawei/data', V3)                      # CSV -> v3/data/*.parquet
    run(f'{PY} prep.py', V2)                                              # CSV -> v2work parquet
    run(f'{PY} v3fold.py val25 f1 f2 f3 f4 f5 f6 val26 eval', V2)         # -> feat_v3/
    run(f'{PY} build_folds.py c25 && {PY} build_folds.py c26', V2)        # -> feat/
    run(f'{PY} extra.py val25 f1 f2 f3 f4 f5 f6 val26 eval', V2)          # -> feat_x/
    run(f'{PY} extra2.py val25 f1 f2 f3 f4 f5 f6 val26 eval', V2)         # -> feat_x2/
    if '--validate' in sys.argv:
        run(f'{PY} combo.py val 25 notime', V2, 'XF=1 XF2=1')
    run(f'{PY} combo.py final 400 3 notime', V2, 'XF=1 XF2=1 OUTP=eval_pred_v3')
    run(f'{PY} write_sub.py out/submission_v3.csv', V2, 'PRED=/workspace/huawei/v2work/eval_pred_v3.parquet')
