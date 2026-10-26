"""v8 test-runner harness: runs solution_v8.py's own pipeline() stage by stage, pinned to 4 cores; records wall / CPU (user+sys of children)
per stage, peak child RSS, and MemAvailable (start / min during stage, sampled every 1 s). Precondition (waits, up to --wait min):
MemAvailable >= 10 GB and no python job with cwd /workspace/huawei/v5. Appends JSON lines to --log.
python timed_run_v8b.py --solution sol/solution_v8.py --version v8 [--raw DIR] [--work DIR]"""
import os, sys, json, time, argparse, subprocess, resource, platform, socket, datetime, threading, glob
ap = argparse.ArgumentParser()
ap.add_argument('--raw', default='/workspace/huawei/data'); ap.add_argument('--work', default=None)
ap.add_argument('--solution', required=True); ap.add_argument('--version', default='v8'); ap.add_argument('--threads', type=int, default=4)
ap.add_argument('--log', default='/workspace/huawei/v8/timings_v8.jsonl'); ap.add_argument('--wait', type=float, default=60); ap.add_argument('--minmem', type=float, default=10.0)
a = ap.parse_args(); T = str(a.threads)
W = os.path.abspath(a.work or f'/workspace/huawei/timed_work_{a.version}'); RAW = os.path.abspath(a.raw)
ns = {'__name__': 'sol'}; exec(open(a.solution).read(), ns)
def memavail():
    for l in open('/proc/meminfo'):
        if l.startswith('MemAvailable'): return int(l.split()[1]) / 1e6
def v5jobs():
    out = []
    for p in glob.glob('/proc/[0-9]*'):
        try:
            if os.readlink(p + '/cwd').startswith('/workspace/huawei/v5'): out.append(p.split('/')[-1])
        except OSError: pass
    return out
t_w = time.time()
while memavail() < a.minmem or v5jobs():
    if time.time() - t_w > a.wait * 60: print('box not quiet: mem %.1f GB, v5 %s' % (memavail(), v5jobs())); sys.exit(3)
    time.sleep(10)
print('quiet: MemAvailable %.1f GB, no v5 jobs (waited %.0fs)' % (memavail(), time.time() - t_w), flush=True)
OUT = os.path.join(W, f'submission_{a.version}.csv'); PY = sys.executable
P = ns['pipeline'](W, RAW, T, PY, OUT); env = ns['thread_env'](T)
PIN = f'taskset -c 0-{a.threads - 1} '
meta = dict(version=a.version, config='v8', solution=os.path.abspath(a.solution), host=socket.gethostname(), cpu=next((l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name')), platform.processor()),
            nproc=os.cpu_count(), threads=a.threads, pinned=PIN.strip(), work=W, run_id=datetime.datetime.now().astimezone().isoformat(timespec='seconds'))
def log(d): open(a.log, 'a').write(json.dumps({**meta, **d, 'ts': datetime.datetime.now().astimezone().isoformat(timespec='seconds')}) + '\n')
mn = [1e9]; stop = [False]
def sampler():
    while not stop[0]: mn[0] = min(mn[0], memavail()); time.sleep(1)
threading.Thread(target=sampler, daemon=True).start()
t_all = time.time(); c_all = resource.getrusage(resource.RUSAGE_CHILDREN); m_all = 1e9; v5seen = False
for name, cmd, sub in P:
    r0 = resource.getrusage(resource.RUSAGE_CHILDREN); m0 = memavail(); mn[0] = m0; t0 = time.time()
    if cmd is None: ns['setup'](W, RAW)
    else:
        print('>>', name, cmd, flush=True)
        p = subprocess.run(PIN + 'env ' + cmd, shell=True, cwd=os.path.join(W, sub), env=env)
        if p.returncode: log(dict(stage=name, error=p.returncode)); sys.exit(p.returncode)
    r1 = resource.getrusage(resource.RUSAGE_CHILDREN); v5 = v5jobs(); v5seen |= bool(v5)
    d = dict(stage=name, wall_s=round(time.time() - t0, 2), cpu_s=round((r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime), 2),
             maxrss_gb=round(r1.ru_maxrss / 1e6, 2), memavail_start_gb=round(m0, 2), memavail_min_gb=round(mn[0], 2), v5_jobs=len(v5))
    m_all = min(m_all, mn[0]); log(d)
    print(f"[stage {name}] wall {d['wall_s']:.1f}s cpu {d['cpu_s']:.1f}s memavail {m0:.1f}->{mn[0]:.1f} GB", flush=True)
stop[0] = True; r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
tot = dict(stage='TOTAL', wall_s=round(time.time() - t_all, 2), cpu_s=round(r1.ru_utime - c_all.ru_utime + r1.ru_stime - c_all.ru_stime, 2), memavail_min_gb=round(m_all, 2),
           v5_seen=v5seen, submission=OUT, sha256=subprocess.run(['sha256sum', OUT], capture_output=True, text=True).stdout.split(' ')[0])
log(tot); print(json.dumps(tot))
