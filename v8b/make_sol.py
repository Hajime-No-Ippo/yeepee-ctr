import os, json, hashlib
B = '/workspace/huawei/v8b'; ns = {}
cand = open('/workspace/huawei/v8/solution_v8_candidate.py').read(); exec(cand.split('def main')[0], ns); SRC0 = ns['SRC']
SRC = {}
for k in SRC0:
    SRC[k] = open(f'{B}/src/{k}').read()
SRC['seq.py'] = open(f'{B}/src/seq.py').read()
SRC['blend_lite.py'] = open(f'{B}/sol/blend_lite.py').read()
# lite config baked in (as measured by timed_run lite_nox2): v2 tables h3..h5 + val26 + eval; feat_x2 optional (not built)
s = SRC['build_folds.py']; a = "specs=[(J20+k*DAY,DAY,f'h{k}') for k in (1,2,3,4,5)]+[(J20+6*DAY"; assert s.count(a) == 1
SRC['build_folds.py'] = s.replace(a, "specs=[(J20+k*DAY,DAY,f'h{k}') for k in (3,4,5)]+[(J20+6*DAY")
s = SRC['cl.py']
s = s.replace("(os.environ.get('XDIR','feat_x'),'feat_x2')", "(os.environ.get('XDIR','feat_x'),)+(('feat_x2',) if os.path.isdir(f'{D}/feat_x2') else ())")
s = s.replace(",f'{D}/feat_x2/{n}.parquet')]", ")+((f'{D}/feat_x2/{n}.parquet',) if os.path.isdir(f'{D}/feat_x2') else ())]")
assert s.count("isdir(f'{D}/feat_x2')") == 2; SRC['cl.py'] = s
del SRC['extra2.py']   # not used by v8 (lite_nox2)
for k, v in SRC.items(): assert '/workspace/huawei/v8b' not in v and '/workspace/huawei/v8/' not in v, k
DOC = '''"""Huawei TechArena 2026 Phase 1 - submission_v8: single self-contained end-to-end pipeline (raw CSV -> features -> model -> CSV).
v8 = v7 lite (laptop budget) + PPMI item-embedding features (ppmi.py) + one sequence model (seq.py) blended as a raw-logit member.
  base : LightGBM binary, lr 0.05, 63 leaves, min_data 200, ff 0.7, bagging 0.8/1, l2 10, max_bin 63; 1 seed (42), 400 rounds
  slow : lr 0.03, ff 0.55, 95 leaves, l2 30, max_bin 63; 1 seed (42), 500 rounds
  both GBDTs train on folds f3..f6 (first event per user in [Ck, Ck+1d), Ck = Jan23..Jan26, stats frozen at Ck), negatives subsampled 0.5 (seed 123)
  seq  : DIN-style target-attention model over each event's own click history (<=20) + slates of the user's last 3 TRAIN events before
         min(event ts, C); item embeddings = frozen PPMI-SVD(64) of train-only click sequences; fitted at C = Jan27 on the last 500k train
         events (ts < C), 3 epochs, batch 256, Adam 2e-3, <= 6000 optimizer steps, seed 0, 4 threads (bounded ~80 s on 4 cores)
  blend: z = mean(base logit, slow logit); z8 = 0.8 z + 0.2 seq_logit; p = sigmoid(z8) in [0, 1].
Features: v3 features (train-only global/item stats + each row's own history), v2 features, similarity extras (extra.py), feat_lr (lrfeat2.py),
feat_tt (ttfeat.py), feat_pp (ppmi.py); time/gap features dropped ('notime'). NO aggregates over evaluation rows: global/item tables come from
training data only; each eval row only contributes its own history to its own user features; within-event (same 10 candidates) z-scores/ranks.
rowid/event_id are never features. v8 speed-ups (same outputs up to float summation order): np.add.at / ufunc.at replaced by sparse
matmuls / sort+reduceat, v2 co-occurrence self-join replaced by sparse B^T B (lower peak memory).
Local validation, same recipe at cutoff Jan25 / Jan26 / Jan23: AUC @@AUC@@.

Usage:  python solution_v8.py --raw DIR_WITH_train.csv_AND_evaluation.csv [--work WORKDIR] [--threads 4]
                                   [--out submission_v8.csv] [--from STAGE] [--to STAGE]
Needs: python 3.11+, polars, numpy, scipy, scikit-learn, lightgbm, torch (CPU) (tested: polars 1.x, lightgbm 4.7). ~10 GB RAM, ~25 GB disk.
Runtime: @@TIME@@
The module sources are embedded below (SRC) and written to WORKDIR at start-up."""
import os, sys, time, json, argparse, subprocess
'''
MAIN = r'''
FOLDS = 'f3 f4 f5 f6 eval'
SLOW = '"learning_rate":0.03,"feature_fraction":0.55,"num_leaves":95,"lambda_l2":30'
LP = '"max_bin":63,"force_col_wise":true'
LR = 'feat_lr2,feat_tt,feat_pp'
FIT = '--negfrac 0.5 --nfolds 4'
def pipeline(W, RAW, T, PY, OUT):
    """ordered (stage, shell command or None for setup, subdir of W)"""
    return [('setup', None, ''), ('prep_v3', f'{PY} prep.py {RAW}', 'v3'), ('prep_v2', f'{PY} prep.py', ''),
            ('feat_v3', f'{PY} v3fold.py {FOLDS}', ''), ('feat_v2', f'{PY} build_folds.py lite', ''),
            ('feat_x', f'{PY} extra.py {FOLDS}', ''), ('feat_lr', f'LR_KS=4,6,8,10,12,16 LR_CON=1 {PY} lrfeat2.py {FOLDS}', ''),
            ('feat_tt', f'TT_THREADS={T} {PY} ttfeat.py {FOLDS}', ''), ('feat_pp', f'{PY} ppmi.py {FOLDS}', ''),
            ('feat_sq', f'SQ_THREADS={T} {PY} seq.py eval', ''),
            ('base', f"{PY} ens.py 27 base --seeds 0 --threads {T} --lr {LR} {FIT} --params '{{{LP}}}'", ''),
            ('slow', f"{PY} ens.py 27 slow --seeds 0 --rounds 500 --threads {T} --lr {LR} {FIT} --params '{{{SLOW},{LP}}}'", ''),
            ('blend', f'{PY} blend_lite.py', ''), ('write', f'PRED={W}/eval_pred.parquet {PY} write_sub.py {OUT}', ''),
            ('validate', f'{PY} validate.py {OUT}', '')]
def setup(W, RAW):
    os.makedirs(os.path.join(W, 'v3'), exist_ok=True); os.makedirs(os.path.join(W, 'ens'), exist_ok=True)
    for k, s in SRC.items():
        open(os.path.join(W, k), 'w').write(s.replace('@@WORK@@', W).replace('@@RAW@@', RAW))
    for f in ('train.csv', 'evaluation.csv'):          # v2 prep / writer read the raw CSVs from WORK
        dst = os.path.join(W, f)
        if not os.path.exists(dst): os.symlink(os.path.join(RAW, f), dst)
def thread_env(T):
    return dict(os.environ, POLARS_MAX_THREADS=T, OMP_NUM_THREADS=T, OPENBLAS_NUM_THREADS=T, MKL_NUM_THREADS=T, NUMEXPR_NUM_THREADS=T,
                TT_THREADS=T, SQ_THREADS=T, RAYON_NUM_THREADS=T, PYTHONUNBUFFERED='1')
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--raw', required=True); ap.add_argument('--work', default='./v8_work')
    ap.add_argument('--threads', type=int, default=4); ap.add_argument('--out', default=None); ap.add_argument('--from', dest='frm', default='setup'); ap.add_argument('--to', default='validate')
    a = ap.parse_args()
    W = os.path.abspath(a.work); RAW = os.path.abspath(a.raw); T = str(a.threads)
    OUT = os.path.abspath(a.out or os.path.join(W, 'submission_v8.csv'))
    env = thread_env(T); PY = sys.executable
    P = pipeline(W, RAW, T, PY, OUT); names = [p[0] for p in P]
    tf = os.path.join(W, 'timings.json'); tim = json.load(open(tf)) if os.path.exists(tf) and a.frm != 'setup' else {}
    t_all = time.time()
    for st, cmd, sub in P[names.index(a.frm):names.index(a.to) + 1]:
        t0 = time.time()
        if cmd is None: setup(W, RAW)
        else: print('>>', cmd, flush=True); subprocess.run(cmd, shell=True, check=True, cwd=os.path.join(W, sub), env=env)
        tim[st] = round(time.time() - t0, 1); print(f'[stage {st}] {tim[st]:.0f}s', flush=True)
        os.makedirs(W, exist_ok=True); json.dump(tim, open(tf, 'w'), indent=1)
    print('TOTAL this run %.1f min; all stages %.1f min' % ((time.time() - t_all) / 60, sum(tim.values()) / 60))
    print('submission:', OUT)
if __name__ == '__main__':
    main()
'''
import sys
auc = sys.argv[1] if len(sys.argv) > 1 else 'TBD'; tm = sys.argv[2] if len(sys.argv) > 2 else 'TBD'
out = DOC.replace('@@AUC@@', auc).replace('@@TIME@@', tm) + '\nSRC = ' + repr(SRC) + '\n' + MAIN
dst = sys.argv[3] if len(sys.argv) > 3 else f'{B}/sol/solution_v8.py'
open(dst, 'w').write(out); compile(out, dst, 'exec'); print(dst, hashlib.sha256(out.encode()).hexdigest(), sorted(SRC))
