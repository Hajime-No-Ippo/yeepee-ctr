"""v8 test-runner harness: runs the v7 solution pipeline stage by stage, pinned to 4 cores, and records
wall and CPU time (user+sys of the child processes) for every stage and the end-to-end total.
Appends one JSON line per stage and one 'TOTAL' line to v2work/out/timings.jsonl.

python timed_run.py --config {full,lite,lite_nox2} [--raw DIR] [--work DIR] [--solution out/solution_v7.py] [--version v7]
  full      = v7 as shipped (f1..f6, base 2 seeds, slow, all feature stages)
  lite      = 1 base seed + slow; both fits: negfrac 0.5, nfolds 4 (f3..f6), max_bin 63, force_col_wise; f1/f2 features skipped
  lite_nox2 = lite and also skips extra2.py (r_/s_ features)
"""
import os, sys, json, time, argparse, subprocess, resource, platform, socket, datetime
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--config', default='lite_nox2', choices=['full', 'lite', 'lite_nox2'])
ap.add_argument('--raw', default='/workspace/huawei/data'); ap.add_argument('--work', default=None)
ap.add_argument('--solution', default=os.path.join(HERE, 'out/solution_v7.py')); ap.add_argument('--version', default='v7')
ap.add_argument('--threads', type=int, default=4); ap.add_argument('--log', default=os.path.join(HERE, 'timings_v8.jsonl'))
a = ap.parse_args(); T = str(a.threads)
W = os.path.abspath(a.work or f'/workspace/huawei/timed_work_{a.version}_{a.config}'); RAW = os.path.abspath(a.raw)
ns = {}; s = open(a.solution).read(); exec(s.split('def main')[0], ns); SRC = dict(ns['SRC'])
LITE = a.config != 'full'
if LITE:   # source patches (in the scratch work dir only)
    SRC['build_folds.py'] = SRC['build_folds.py'].replace("specs=[(J20+k*DAY,DAY,f'h{k}') for k in (1,2,3,4,5)]", "specs=[(J20+k*DAY,DAY,f'h{k}') for k in (3,4,5)]")
    assert "for k in (3,4,5)]+[(J20+6*DAY" in SRC['build_folds.py']
    SRC['blend_lite.py'] = SRC['blend_lite.py'].replace("NS = {'base': 2, 'slow': 1}", "NS = {'base': 1, 'slow': 1}")
    SRC['cl.py'] = SRC['cl.py'].replace("(os.environ.get('XDIR','feat_x'),'feat_x2')", "(os.environ.get('XDIR','feat_x'),)+(('feat_x2',) if os.path.isdir(f'{D}/feat_x2') else ())")
    SRC['cl.py'] = SRC['cl.py'].replace(",f'{D}/feat_x2/{n}.parquet')]", ")+((f'{D}/feat_x2/{n}.parquet',) if os.path.isdir(f'{D}/feat_x2') else ())]")
    assert SRC['cl.py'].count("isdir(f'{D}/feat_x2')") == 2
LRX = ',feat_pp' if 'ppmi.py' in SRC else ''
FOLDS = 'f3 f4 f5 f6 eval' if LITE else 'f1 f2 f3 f4 f5 f6 eval'
SLOW = '{"learning_rate":0.03,"feature_fraction":0.55,"num_leaves":95,"lambda_l2":30}'
LP = '"max_bin":63,"force_col_wise":true'
FIT = " --negfrac 0.5 --nfolds 4" if LITE else ''
BP = f"--params '{{{LP}}}'" if LITE else ''
SP = f"--params '{SLOW[:-1]},{LP}}}'" if LITE else f"--params '{SLOW}'"
env = dict(os.environ, POLARS_MAX_THREADS=T, OMP_NUM_THREADS=T, OPENBLAS_NUM_THREADS=T, MKL_NUM_THREADS=T, NUMEXPR_NUM_THREADS=T,
           TT_THREADS=T, RAYON_NUM_THREADS=T, PYTHONUNBUFFERED='1')
PIN = f'taskset -c 0-{a.threads - 1} ' if subprocess.run('command -v taskset', shell=True, capture_output=True).returncode == 0 else ''
PY = sys.executable
def setup():
    os.makedirs(os.path.join(W, 'v3'), exist_ok=True); os.makedirs(os.path.join(W, 'ens'), exist_ok=True)
    for k, v in SRC.items(): open(os.path.join(W, k), 'w').write(v.replace('@@WORK@@', W).replace('@@RAW@@', RAW))
    for f in ('train.csv', 'evaluation.csv'):
        if not os.path.exists(os.path.join(W, f)): os.symlink(os.path.join(RAW, f), os.path.join(W, f))
OUT = os.path.join(W, f'submission_{a.version}_{a.config}.csv')
ST = [('setup', None), ('load_prep_v3', (f'{PY} prep.py {RAW}', 'v3')), ('load_prep_v2', (f'{PY} prep.py', '')),
      ('feat_v3', (f'{PY} v3fold.py {FOLDS}', '')), ('feat_v2', (f'{PY} build_folds.py lite', '')),
      ('feat_x', (f'{PY} extra.py {FOLDS}', ''))]
if a.config != 'lite_nox2': ST.append(('feat_x2', (f'{PY} extra2.py {FOLDS}', '')))
ST += [('feat_lr', (f'LR_KS=4,6,8,10,12,16 LR_CON=1 {PY} lrfeat2.py {FOLDS}', '')), ('feat_tt', (f'{PY} ttfeat.py {FOLDS}', '')),
       *([('feat_pp', (f'{PY} ppmi.py {FOLDS}', ''))] if 'ppmi.py' in SRC else []),
       ('fit_predict_base', (f"{PY} ens.py 27 base --seeds {'0' if LITE else '0,1'} --threads {T} --lr feat_lr2,feat_tt{LRX}{FIT} {BP}", '')),
       ('fit_predict_slow', (f"{PY} ens.py 27 slow --seeds 0 --rounds 500 --threads {T} --lr feat_lr2,feat_tt{LRX}{FIT} {SP}", '')),
       ('blend', (f'{PY} blend_lite.py', '')), ('write', (f'PRED={W}/eval_pred.parquet {PY} write_sub.py {OUT}', '')),
       ('validate', (f'{PY} validate.py {OUT}', ''))]
meta = dict(version=a.version, config=a.config, host=socket.gethostname(), cpu=next((l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name')), platform.processor()),
            nproc=os.cpu_count(), threads=a.threads, pinned=PIN.strip() or None, work=W, run_id=datetime.datetime.now().astimezone().isoformat(timespec='seconds'))
def log(d): os.makedirs(os.path.dirname(a.log), exist_ok=True); open(a.log, 'a').write(json.dumps({**meta, **d, 'ts': datetime.datetime.now().astimezone().isoformat(timespec='seconds')}) + '\n')
t_all = time.time(); c_all = resource.getrusage(resource.RUSAGE_CHILDREN); rows = []
for name, cmd in ST:
    r0 = resource.getrusage(resource.RUSAGE_CHILDREN); s0 = resource.getrusage(resource.RUSAGE_SELF); t0 = time.time()
    if cmd is None: setup()
    else:
        c, sub = cmd; print('>>', name, c, flush=True)
        p = subprocess.run(PIN + 'env ' + c, shell=True, cwd=os.path.join(W, sub), env=env)
        if p.returncode: log(dict(stage=name, error=p.returncode)); sys.exit(p.returncode)
    r1 = resource.getrusage(resource.RUSAGE_CHILDREN); s1 = resource.getrusage(resource.RUSAGE_SELF)
    cpu = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime) + (s1.ru_utime - s0.ru_utime) + (s1.ru_stime - s0.ru_stime)
    d = dict(stage=name, wall_s=round(time.time() - t0, 2), cpu_s=round(cpu, 2), maxrss_gb=round(r1.ru_maxrss / 1e6, 2)); rows.append(d); log(d)
    print(f"[stage {name}] wall {d['wall_s']:.1f}s cpu {d['cpu_s']:.1f}s", flush=True)
r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
tot = dict(stage='TOTAL', wall_s=round(time.time() - t_all, 2), cpu_s=round(r1.ru_utime - c_all.ru_utime + r1.ru_stime - c_all.ru_stime, 2), submission=OUT,
           sha256=subprocess.run(['sha256sum', OUT], capture_output=True, text=True).stdout.split(' ')[0])
log(tot); print(json.dumps(tot))
