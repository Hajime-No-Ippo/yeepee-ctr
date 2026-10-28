"""seqrun.py NAMES... : multi-seed sequence-model stage. Runs seq11.py once per seed (SQ_SEED = 0..V12_SQ_SEEDS-1) with at most V12_SQ_PAR runs
at the same time, SQ_THREADS torch threads each, then averages the seeds into feat_sqC/ (avg_sq.py).
Seeds 0-7: base config (== seq10); seeds 8-10: SQ_H=4 SQ_HID=128 (4 heads, hidden 128) for ensemble diversity.
Env: V12_SQ_SEEDS (default 11), V12_SQ_PAR (default min(3, threads//2)), V12_SQ_THREADS (torch threads per run, default threads//par)."""
import os, sys, time, subprocess
W = '@@WORK@@'; names = sys.argv[1:]
NS = int(os.environ.get('V12_SQ_SEEDS', 11)); T = int(os.environ.get('V12_THREADS', 8))
PAR = int(os.environ.get('V12_SQ_PAR', max(1, min(3, T // 2)))); TH = int(os.environ.get('V12_SQ_THREADS', max(1, T // PAR)))
print(f'seqrun: {NS} seeds, {PAR} concurrent, {TH} threads each, tables {names}', flush=True)
assert NS >= 1
def cfg(s): return {'SQ_H': '4', 'SQ_HID': '128'} if s >= 8 else {}
todo = list(range(NS)); run = {}; t0 = time.time(); done = []
os.makedirs(f'{W}/msC/logs', exist_ok=True)
def finished(s): return all(os.path.exists(f'{W}/msC/s{s}/{n}.parquet') for n in names) and os.path.exists(f'{W}/msC/s{s}/DONE')
todo = [s for s in todo if not finished(s)]; print('seeds to run', todo, flush=True)
while todo or run:
    while todo and len(run) < PAR:
        s = todo.pop(0); od = f'{W}/msC/s{s}'; os.makedirs(od, exist_ok=True)
        env = dict(os.environ, SQ_SEED=str(s), SQ_THREADS=str(TH), SQ_OUT=od, SQ_WORK=W, SQ_NTR='500000', POLARS_MAX_THREADS=str(TH), OMP_NUM_THREADS=str(TH), OPENBLAS_NUM_THREADS=str(TH), MKL_NUM_THREADS=str(TH), PYTHONUNBUFFERED='1', **cfg(s))
        lf = open(f'{W}/msC/logs/s{s}.log', 'w')
        run[s] = (subprocess.Popen([sys.executable, 'seq11.py'] + names, cwd=W, env=env, stdout=lf, stderr=subprocess.STDOUT), time.time()); print(f'[{time.time() - t0:.0f}s] start seed {s}', flush=True)
        time.sleep(15)   # stagger the memory-heavy start-up
    time.sleep(3)
    for s, (p, ts) in list(run.items()):
        r = p.poll()
        if r is None: continue
        del run[s]
        if r != 0:
            for q, _ in run.values(): q.kill()
            sys.exit(f'seq seed {s} failed (rc {r}); see {W}/msC/logs/s{s}.log')
        open(f'{W}/msC/s{s}/DONE', 'w').write('ok'); print(f'[{time.time() - t0:.0f}s] seed {s} done in {time.time() - ts:.0f}s', flush=True)
subprocess.run([sys.executable, 'avg_sq.py', f'{W}/feat_sqC'] + [f'{W}/msC/s{s}' for s in range(NS)], cwd=W, check=True)
print(f'seqrun done {time.time() - t0:.0f}s', flush=True)
