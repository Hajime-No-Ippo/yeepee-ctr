#!/usr/bin/env python3
"""v17run: rebuild the Huawei TechArena 2026 CTR submission 'v17cooc' (Team Yeepee) from the raw CSVs.

 v17cooc = v12 recipe (435 columns: v3/v2/extra/lr2/tt/pp + DIN sequence stack sqC + DW dw features; LightGBM dwbig + dw_s + 3x MLP, logit blend 1:1:2)
           + 25 item co-occurrence (PPMI -> SVD) columns  coA_* (variant A, train-only graph)  or  coB_* (variant B = TRANSDUCTIVE: the history of the evaluation
           events is added to the co-occurrence graph; no evaluation labels exist or are used).  -> 460 columns.

 python v17run.py [--data DATA_DIR] [--work WORK_DIR] [--variant A|B|both] [--cut 27|25|both] [--threads 8] [--from STAGE] [--to STAGE] [--p33 P33_PATH] [--force] [--list]
   DATA_DIR  directory holding train.csv and evaluation.csv   (env DATA_DIR, default <this dir>/data)
   WORK_DIR  scratch / output directory                         (env WORK_DIR, default <this dir>/work);  cut 27 -> WORK_DIR/c27, cut 25 -> WORK_DIR/c25
   --cut 27  builds the submission(s)  WORK_DIR/c27/out/submission_v17cooc_{A_trainonly,B_TRANSDUCTIVE}.csv(.zip)
   --cut 25  local validation (folds f1..f4, scored on val25 with the train labels) -> WORK_DIR/c25/local_auc_c25_{A,B}.json
   P33_PATH  optional parquet (rowid, p33_rank) of the teammate's p33 model; if given, the 0.8*rank(ours)+0.2*rank(p33) probe is written too.
 Stages are resumable: every finished stage writes WORK_DIR/c<cut>/.done/<stage>; rerun the same command to continue (--force redoes, --from/--to select a range).
 NEVER uploads anything anywhere. Output CSVs are what you would submit; uploading is a manual step."""
import os, sys, time, json, argparse, subprocess, shutil
HERE = os.path.dirname(os.path.abspath(__file__)); SRC = os.path.join(HERE, 'src')
SLOW = '"learning_rate":0.03,"feature_fraction":0.55,"num_leaves":95,"lambda_l2":30'
BIG = '"learning_rate":0.015,"num_leaves":127,"feature_fraction":0.5,"lambda_l2":30,"min_data_in_leaf":200,"max_bin":255,"force_col_wise":true'
LP = '"max_bin":63,"force_col_wise":true'
LR = 'feat_lr2,feat_tt,feat_pp,feat_sqC,feat_dw'
DROP = "'^sq_(pdot|pcos|mdot|mcos)$'"
NAME = {'A': 'submission_v17cooc_A_trainonly', 'B': 'submission_v17cooc_B_TRANSDUCTIVE'}

def pipeline(W, T, PY, cut, variants, p33):
    FOLDS = 'f3 f4 f5 f6 eval' if cut == 27 else 'f1 f2 f3 f4 val25'
    V2 = 'lite' if cut == 27 else 'c25'
    P = [('setup', None, ''), ('prep_v3', f'{PY} prep.py $RAW', 'v3'), ('prep_v2', f'{PY} prep.py', ''),
         ('feat_v3', f'{PY} v3fold.py {FOLDS}', ''), ('feat_v2', f'{PY} build_folds.py {V2}', ''),
         ('feat_x', f'{PY} extra.py {FOLDS}', ''), ('feat_lr', f'LR_KS=4,6,8,10,12,16 LR_CON=1 {PY} lrfeat2.py {FOLDS}', ''),
         ('feat_tt', f'TT_THREADS={T} {PY} ttfeat.py {FOLDS}', ''), ('feat_pp', f'{PY} ppmi.py {FOLDS}', ''),
         ('feat_sq', f'V12_THREADS={T} {PY} seqrun.py {FOLDS}', ''), ('feat_dw', f'{PY} dw/mkdw.py {FOLDS}', '')]
    for v in variants: P.append((f'cooc_{v}', f'{PY} cooc/mkcooc.py {v} {FOLDS}', ''))
    P.append(('cooc_sel', f'{PY} cooc/mksel.py', ''))
    for v in variants:
        L = f'{LR},feat_{v}p'
        P += [(f'{v}_big', f"{PY} ens.py {cut} {v}big --seeds 0 --rounds 1000 --save 1000 --threads {T} --lr {L} --nfolds 4 --drop {DROP} --params '{{{BIG}}}'", ''),
              (f'{v}_s', f"{PY} ens.py {cut} {v}_s --seeds 0 --rounds 500 --save 500 --threads {T} --lr {L} --nfolds 4 --negfrac 0.5 --drop {DROP} --params '{{{SLOW},{LP}}}'", ''),
              *[(f'{v}_mlp{s}', f'MLP_THREADS={T} {PY} mlp12.py {cut} {v}mlp --ep 4 --seed {s} --lr {L}', '') for s in range(3)],
              (f'{v}_blend', f'{PY} cooc/blend_cooc.py {cut} {v}', '')]
        if cut == 27:
            o = f'out/{NAME[v]}.csv'
            P += [(f'{v}_write', f'mkdir -p out && PRED={W}/eval_pred_{v}.parquet {PY} write_sub.py {o}', ''), (f'{v}_validate', f'{PY} validate.py {o}', '')]
            if p33: P.append((f'{v}_probe', f'{PY} cooc/mkprobe.py {o} {p33} out/probe_v17cooc_{NAME[v][len("submission_v17cooc_"):]}_p33_w80.csv', ''))
    return P

def setup(W, RAW):
    for r, _, fs in os.walk(SRC):
        for f in fs:
            if not f.endswith(('.py', '.sh')) or '__pycache__' in r: continue
            p = os.path.join(r, f); k = os.path.relpath(p, SRC); dst = os.path.join(W, k); os.makedirs(os.path.dirname(dst), exist_ok=True)
            open(dst, 'w').write(open(p).read().replace('@@WORK@@', W).replace('@@RAW@@', RAW))
    for d in ('ens', 'logs', 'out'): os.makedirs(os.path.join(W, d), exist_ok=True)
    for f in ('train.csv', 'evaluation.csv'):          # the v2 prep / writer read the raw CSVs from WORK
        dst = os.path.join(W, f)
        if not os.path.lexists(dst): os.symlink(os.path.join(RAW, f), dst)

def run_cut(a, cut, variants):
    W = os.path.join(os.path.abspath(a.work), f'c{cut}'); RAW = os.path.abspath(a.data); os.makedirs(W, exist_ok=True)
    T = str(a.threads); PY = sys.executable
    env = dict(os.environ, POLARS_MAX_THREADS=T, OMP_NUM_THREADS=T, OPENBLAS_NUM_THREADS=T, MKL_NUM_THREADS=T, NUMEXPR_NUM_THREADS=T,
               TT_THREADS=T, SQ_THREADS=T, RAYON_NUM_THREADS=T, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
    P = pipeline(W, T, PY, cut, variants, a.p33); names = [p[0] for p in P]
    if a.list: print(f'cut {cut}:', ' '.join(names)); return
    frm = a.frm or names[0]; to = a.to or names[-1]; assert frm in names and to in names, names
    os.makedirs(f'{W}/.done', exist_ok=True); tf = f'{W}/timings.json'; tim = json.load(open(tf)) if os.path.exists(tf) else {}
    t_all = time.time()
    for st, cmd, sub in P[names.index(frm):names.index(to) + 1]:
        mk = f'{W}/.done/{st}'
        if st != 'setup' and os.path.exists(mk) and not a.force: print(f'[skip {st}: done]', flush=True); continue
        if st == 'setup': setup(W, RAW); open(mk, 'w').write('ok')
        else:
            if not os.path.exists(f'{W}/prep.py'): setup(W, RAW)
            t0 = time.time(); c = cmd.replace('$RAW', RAW); print(f'[cut {cut}] >> {c}', flush=True)
            subprocess.run(c, shell=True, check=True, cwd=os.path.join(W, sub), env=env)
            tim[st] = round(time.time() - t0, 1); print(f'[stage {st}] {tim[st]:.0f}s', flush=True)
            json.dump(tim, open(tf, 'w'), indent=1); open(mk, 'w').write('ok')
    print(f'cut {cut}: TOTAL this run {(time.time()-t_all)/60:.1f} min; all stages {sum(tim.values())/60:.1f} min')
    if cut == 27 and not a.list: print('submissions in', f'{W}/out')

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default=os.environ.get('DATA_DIR', os.path.join(HERE, 'data'))); ap.add_argument('--work', default=os.environ.get('WORK_DIR', os.path.join(HERE, 'work')))
    ap.add_argument('--variant', default='both', choices=('A', 'B', 'both')); ap.add_argument('--cut', default='27', choices=('27', '25', 'both'))
    ap.add_argument('--threads', type=int, default=int(os.environ.get('THREADS', 8))); ap.add_argument('--from', dest='frm', default=None); ap.add_argument('--to', default=None)
    ap.add_argument('--p33', default=os.environ.get('P33_PATH')); ap.add_argument('--force', action='store_true'); ap.add_argument('--list', action='store_true')
    a = ap.parse_args()
    variants = ['A', 'B'] if a.variant == 'both' else [a.variant]
    for f in ('train.csv', 'evaluation.csv'):
        if not a.list and not os.path.exists(os.path.join(a.data, f)): sys.exit(f'missing {os.path.join(a.data, f)} (set DATA_DIR / --data)')
    if 'B' in variants: print('NOTE: variant B is TRANSDUCTIVE (evaluation-set history entries are added to the item co-occurrence graph; no labels used).', flush=True)
    for cut in ([27, 25] if a.cut == 'both' else [int(a.cut)]): run_cut(a, cut, variants)
if __name__ == '__main__': main()
