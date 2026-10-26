STAGES = ['setup', 'prep_v3', 'prep_v2', 'feat_v3', 'feat_v2', 'feat_x', 'feat_x2', 'feat_lr', 'base', 'slow', 'blend', 'write', 'validate']
FOLDS = 'f1 f2 f3 f4 f5 f6 eval'
SLOW = '{"learning_rate":0.03,"feature_fraction":0.55,"num_leaves":95,"lambda_l2":30}'
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--raw', required=True); ap.add_argument('--work', default='./v6_work')
    ap.add_argument('--threads', type=int, default=4); ap.add_argument('--out', default=None); ap.add_argument('--from', dest='frm', default='setup'); ap.add_argument('--to', default='validate')
    a = ap.parse_args()
    W = os.path.abspath(a.work); RAW = os.path.abspath(a.raw); T = str(a.threads)
    OUT = os.path.abspath(a.out or os.path.join(W, 'submission_v6.csv'))
    env = dict(os.environ, POLARS_MAX_THREADS=T, OMP_NUM_THREADS=T, OPENBLAS_NUM_THREADS=T, MKL_NUM_THREADS=T, NUMEXPR_NUM_THREADS=T, PYTHONUNBUFFERED='1')
    PY = sys.executable
    def sh(cmd, cwd=W): print('>>', cmd, flush=True); subprocess.run(cmd, shell=True, check=True, cwd=cwd, env=env)
    def setup():
        os.makedirs(os.path.join(W, 'v3'), exist_ok=True); os.makedirs(os.path.join(W, 'ens'), exist_ok=True)
        for k, s in SRC.items():
            open(os.path.join(W, k), 'w').write(s.replace('@@WORK@@', W).replace('@@RAW@@', RAW))
        for f in ('train.csv', 'evaluation.csv'):          # v2 prep / writer read the raw CSVs from WORK
            dst = os.path.join(W, f)
            if not os.path.exists(dst): os.symlink(os.path.join(RAW, f), dst)
    run = {'setup': setup,
           'prep_v3': lambda: sh(f'{PY} prep.py {RAW}', os.path.join(W, 'v3')),
           'prep_v2': lambda: sh(f'{PY} prep.py'),
           'feat_v3': lambda: sh(f'{PY} v3fold.py {FOLDS}'),
           'feat_v2': lambda: sh(f'{PY} build_folds.py lite'),
           'feat_x': lambda: sh(f'{PY} extra.py {FOLDS}'),
           'feat_x2': lambda: sh(f'{PY} extra2.py {FOLDS}'),
           'feat_lr': lambda: sh(f'LR_KS=4,6,8,10,12,16 LR_CON=1 {PY} lrfeat2.py {FOLDS}'),
           'base': lambda: sh(f'{PY} ens.py 27 base --seeds 0,1 --threads {T} --lr feat_lr2'),
           'slow': lambda: sh(f"{PY} ens.py 27 slow --seeds 0 --rounds 500 --threads {T} --lr feat_lr2 --params '{SLOW}'"),
           'blend': lambda: sh(f'{PY} blend_lite.py'),
           'write': lambda: sh(f'PRED={W}/eval_pred.parquet {PY} write_sub.py {OUT}'),
           'validate': lambda: sh(f'{PY} validate.py {OUT}')}
    tf = os.path.join(W, 'timings.json'); tim = json.load(open(tf)) if os.path.exists(tf) and a.frm != 'setup' else {}
    t_all = time.time()
    for st in STAGES[STAGES.index(a.frm):STAGES.index(a.to) + 1]:
        t0 = time.time(); run[st](); tim[st] = round(time.time() - t0, 1)
        print(f'[stage {st}] {tim[st]:.0f}s', flush=True)
        os.makedirs(W, exist_ok=True); json.dump(tim, open(tf, 'w'), indent=1)
    print('TOTAL this run %.1f min; all stages %.1f min' % ((time.time() - t_all) / 60, sum(tim.values()) / 60))
    print('submission:', OUT)
if __name__ == '__main__':
    main()
