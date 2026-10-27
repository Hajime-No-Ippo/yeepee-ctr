"""make_sol10.py MAXSTEPS -> sol/solution_v10.py (from a COPY of v8b/sol/solution_v8.py; v8 file untouched)."""
import ast, sys, re
MS = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
s = open('/workspace/huawei/v8b/sol/solution_v8.py').read(); i = s.index("SRC = "); j = s.index('\n', i)
SRC = ast.literal_eval(s[i + 6:j])
q = open('/workspace/huawei/v10/seq10.py').read()
q = q.replace("f'{D}/feat_sq'", "f'{D}/feat_sq10'").replace("os.environ.get('SQ_MAXSTEPS', 6000)", f"os.environ.get('SQ_MAXSTEPS', {MS})")
assert "'@@WORK@@'" in q and f"SQ_MAXSTEPS', {MS})" in q and 'feat_sq10' in q
old = "df = pl.DataFrame(cols).with_columns(pl.exclude('rowid').cast(pl.Float32))"
assert old in q
q = q.replace(old, "for n_ in ['sq_pdot', 'sq_pcos', 'sq_mdot', 'sq_mcos']: cols.pop(n_)  # v10B: fit-specific scale drifts train->eval (KS .09-.16); keep _dm versions\n    " + old)
SRC['seq10.py'] = q
SRC['bgseq.sh'] = '''# v10: start the sequence-model stage in the background (it only needs v3/data from prep_v3) pinned to the last core with 1 torch thread, while the polars feature stages run on the other cores
rm -f seq10.rc seq10.log
( taskset -c "$2" "$1" seq10.py f3 f4 f5 f6 eval > seq10.log 2>&1; echo $? > seq10.rc ) < /dev/null > /dev/null 2>&1 &
echo "seq10 started in background on core $2 (SQ_THREADS=$SQ_THREADS)"
'''
SRC['waitseq.sh'] = '''# v10: block until the background sequence stage finishes; fail if it failed
while [ ! -f seq10.rc ]; do sleep 1; done
grep -v ' ep' seq10.log; exit $(cat seq10.rc)
'''
b = SRC['blend_lite.py']
b = b.replace('"""v8 final blend', '"""v10 final blend (no sequence post-blend: the sequence model is stacked into both GBDTs via feat_sq10)\nv8 final blend').replace('WSQ = 0.2', 'WSQ = 0.0')
b = b.replace("'feat_sq/eval.parquet'", "'feat_sq10/eval.parquet'"); assert 'WSQ = 0.0' in b and 'feat_sq10/eval' in b; SRC['blend_lite.py'] = b
head = s[:i]; tail = s[j:]
head = head.replace('"""Huawei TechArena 2026 Phase 1 - submission_v8:', '"""Huawei TechArena 2026 Phase 1 - submission_v10 (v8 + sequence model STACKED before LightGBM):\n'
  f'v10 changes vs v8: seq10.py (same DIN-style model, <= {MS} optimizer steps) is refit at every fold cutoff Ck = Jan23..Jan26 (train events ts < Ck only)\n'
  '  and at C = Jan27 for eval; its outputs on each fold\'s rows (temporally out-of-sample: rows are >= Ck) become 15 LightGBM features (feat_sq10:\n'
  '  logit, within-event z/demeaned/rank/gap, attention max/null/mean/entropy, event-demeaned pooled-user/mean-key vs candidate cos, PPMI history cos (+demeaned));\n'
  '  the 0.2 raw-logit post-blend is removed (z = mean(base, slow)). The sequence stage runs in the background from right after prep_v3\n  (it only reads v3/data) overlapping the polars feature stages; feat_sq_wait joins it before the GBDTs.\n'
  'Original v8 docstring follows.\n')
tail = tail.replace("LR = 'feat_lr2,feat_tt,feat_pp'", "LR = 'feat_lr2,feat_tt,feat_pp,feat_sq10'")
old = "('feat_sq', f'SQ_THREADS={T} {PY} seq.py eval', ''),"; assert old in tail; tail = tail.replace(old, "('feat_sq_wait', 'bash waitseq.sh', ''),")
old = "('prep_v3', f'{PY} prep.py {RAW}', 'v3'),"; assert old in tail
tail = tail.replace(old, old + " ('feat_sq_start', f'SQ_THREADS={SQT} bash bgseq.sh {PY}', ''),")
tail = tail.replace("LR = '", f"SQT = int(os.environ.get('V10_SQ_THREADS', 3))   # threads for the overlapped sequence stage\nLR = '", 1)
tail = tail.replace("submission_v8.csv')", "submission_v10.csv')").replace("'./v8_work'", "'./v10_work'")
assert 'feat_sq10' in tail and 'waitseq' in tail and 'bgseq' in tail
PIPE2 = """
def pipeline(W, RAW, T, PY, OUT):
    \"\"\"v10: the sequence stage (1 torch thread) runs in the background on the last core from feat_sq_start to feat_sq_wait;
    stages in between are re-pinned to the remaining cores with thread counts = cores - 1 (no oversubscription).\"\"\"
    P = _pipeline0(W, RAW, T, PY, OUT); n = int(T); assert n >= 2
    lo = f'0-{n - 2}' if n > 2 else '0'; t2 = str(n - 1); out = []; ov = False
    TV = ('POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'RAYON_NUM_THREADS', 'NUMBA_NUM_THREADS')
    for name, cmd, sub in P:
        if name == 'feat_sq_start': cmd = f'SQ_THREADS=1 bash bgseq.sh {PY} {n - 1}'; ov = True
        elif name == 'feat_sq_wait': ov = False
        elif ov and cmd:
            toks = cmd.split(' '); k = 0
            while '=' in toks[k] and not toks[k].startswith('-'): k += 1
            own = ' '.join(toks[:k]).replace(f'TT_THREADS={T}', f'TT_THREADS={t2}')
            cmd = ' '.join(x for x in (' '.join(f'{v}={t2}' for v in TV), own, f'taskset -c {lo}', ' '.join(toks[k:])) if x)
        out.append((name, cmd, sub))
    return out
"""
assert 'def pipeline(W, RAW, T, PY, OUT):' in tail
tail = tail.replace('def pipeline(W, RAW, T, PY, OUT):', 'def _pipeline0(W, RAW, T, PY, OUT):', 1)
k = tail.index('def setup('); tail = tail[:k] + PIPE2.lstrip('\n') + tail[k:]
open('/workspace/huawei/v10/sol/solution_v10.py', 'w').write(head + 'SRC = ' + repr(SRC) + tail)
print('ok', MS)
