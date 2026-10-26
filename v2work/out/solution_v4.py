"""Huawei TechArena 2026 Phase 1 - submission_v4 pipeline (driver). GBDT-only ensemble on top of the v3 feature pipeline.
NO aggregates over evaluation rows: all global/item-level stats use training data only; each eval row's own history is used
only for that user's own features. Within-event (same 10 candidates) z-scores kept. rowid/event_id never used as features.
Components (LightGBM binary, raw-score/logit outputs, seed-averaged; base params lr 0.05, 63 leaves, min_data 200, ff 0.7,
bagging 0.8/1, l2 10):
  base  : v3 features ('notime'), 7 seeds, 400 rounds, trained on f1..f6 (first event per user in [Ck, Ck+1d), Ck=Jan21..26)
  wide  : ff 0.4, 127 leaves, min_data 400, 2 seeds, 400 rounds, f1..f6
  slow  : lr 0.03, ff 0.55, 95 leaves, l2 30, 1 seed, 500 rounds, f1..f6
  estrc : + tm_stale (days since train end) + repeat-click/co-vis features (rcov.py, train-only tables), 2 seeds, 400 rounds,
          trained on eval-like multi-day folds E1..E6 ([Ck, Jan27), stats frozen at Ck)
Blend: equal-weight mean of the 4 component logits, sigmoid -> probability (nov2 component and sequence model excluded).
Local validation like-for-like (same recipe at cutoff Jan25 / Jan26): 0.77895 / 0.79454 vs v3-recipe reference 0.77852 / 0.79415.
Sources: solution_v4_src.zip."""
import subprocess, sys
PY = sys.executable
V2 = '/workspace/huawei/v2work'
def run(cmd, cwd=V2, env=''): print('>>', cmd); subprocess.run(f'{env} {cmd}', shell=True, check=True, cwd=cwd)
if __name__ == '__main__':
    # 1) features exactly as v3 (see solution_v3.py): v3/prep.py, v2work prep.py, v3fold.py, build_folds.py, extra.py, extra2.py
    #    for folds f1..f6, E1..E6, val25, val26, eval
    #    (E1..E6 built with the same scripts: v3fold.py / extra.py / extra2.py E1..E6, build_folds.py; they already exist in
    #    feat_v3/, feat/, feat_x/, feat_x2/). This driver starts from those feature parquets.
    run(f'{PY} rcov.py E1 E2 E3 E4 E5 E6 eval')                                  # -> feat_rc/ (train-only tables)
    run(f'{PY} ens.py 27 base --seeds 0,1,2,3,4,5,6 --threads 4')
    run(f'{PY} ens.py 27 wide --seeds 0,1 --threads 4 --params \'{{"feature_fraction":0.4,"num_leaves":127,"min_data_in_leaf":400}}\'')
    run(f'{PY} ens.py 27 slow --seeds 0 --rounds 500 --threads 4 --params \'{{"learning_rate":0.03,"feature_fraction":0.55,"num_leaves":95,"lambda_l2":30}}\'')
    run(f'{PY} ens.py 27 estrc --seeds 0,1 --folds e --stale --rc --threads 4')
    run(f'{PY} blend27.py')                                                      # -> eval_pred_v4.parquet
    run(f'{PY} write_sub.py out/submission_v4.csv', env='PRED=/workspace/huawei/v2work/eval_pred_v4.parquet')
    run(f'{PY} validate.py out/submission_v4.csv')
