"""Huawei TechArena 2026 Phase 1 - submission_v2 pipeline (driver).
Model: LightGBM binary on v3 features (/workspace/huawei/v3: features.py, split.py, prep.py, build_feats.py)
       + v2 features (v2work: prep.py, feats.py, build_folds.py), joined on rowid; time/gap features dropped ('notime').
Training rows: first event per user in [Ck, Ck+1d) for cutoffs Jan21..Jan26 (6 folds, all stats frozen at each Ck).
Eval features: stats frozen at Jan 27 00:00 UTC. 450 rounds, lr 0.05, 63 leaves, 3 seeds averaged (plain mean, no per-event normalisation).
Local validation (cutoff Jan25, train on Jan21..24 folds): AUC 0.7620 (v3-only 0.7561, v2-only 0.7364, starter 0.611).
Sources for every step are in solution_v2_src.zip (unzip next to this file, keep the folder layout).
"""
import subprocess, sys
PY = sys.executable
V3 = '/workspace/huawei/v3'; V2 = '/workspace/huawei/v2work'
def run(cmd, cwd): print('>>', cmd); subprocess.run(cmd, shell=True, check=True, cwd=cwd)
if __name__ == '__main__':
    run(f'{PY} prep.py /workspace/huawei/data', V3)            # CSV -> v3/data/*.parquet
    run(f'{PY} build_feats.py all', V3)                         # v3/feat/{f1..f6,val25,val26,eval}.parquet
    run(f'{PY} prep.py', V2)                                    # CSV -> v2work/*_events/*_rows parquet
    run(f'{PY} build_folds.py c25 && {PY} build_folds.py c26', V2)  # v2work/feat/{h1..h5,val25,val26,eval}.parquet
    if '--validate' in sys.argv:
        run(f'{PY} combo.py val 25 notime', V2)
    run(f'{PY} combo.py final 450 3 notime', V2)                # -> eval_pred.parquet
    run(f'{PY} write_sub.py out/submission_v2.csv', V2)         # write + validate + zip
