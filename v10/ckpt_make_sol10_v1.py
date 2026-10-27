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
b = SRC['blend_lite.py']
b = b.replace('"""v8 final blend', '"""v10 final blend (no sequence post-blend: the sequence model is stacked into both GBDTs via feat_sq10)\nv8 final blend').replace('WSQ = 0.2', 'WSQ = 0.0')
b = b.replace("'feat_sq/eval.parquet'", "'feat_sq10/eval.parquet'"); assert 'WSQ = 0.0' in b and 'feat_sq10/eval' in b; SRC['blend_lite.py'] = b
head = s[:i]; tail = s[j:]
head = head.replace('"""Huawei TechArena 2026 Phase 1 - submission_v8:', '"""Huawei TechArena 2026 Phase 1 - submission_v10 (v8 + sequence model STACKED before LightGBM):\n'
  f'v10 changes vs v8: seq10.py (same DIN-style model, <= {MS} optimizer steps) is refit at every fold cutoff Ck = Jan23..Jan26 (train events ts < Ck only)\n'
  '  and at C = Jan27 for eval; its outputs on each fold\'s rows (temporally out-of-sample: rows are >= Ck) become 15 LightGBM features (feat_sq10:\n'
  '  logit, within-event z/demeaned/rank/gap, attention max/null/mean/entropy, event-demeaned pooled-user/mean-key vs candidate cos, PPMI history cos (+demeaned));\n'
  '  the 0.2 raw-logit post-blend is removed (z = mean(base, slow)).\n'
  'Original v8 docstring follows.\n')
tail = tail.replace("LR = 'feat_lr2,feat_tt,feat_pp'", "LR = 'feat_lr2,feat_tt,feat_pp,feat_sq10'")
tail = tail.replace("('feat_sq', f'SQ_THREADS={T} {PY} seq.py eval', '')", "('feat_sq', f'SQ_THREADS={T} {PY} seq10.py f3 f4 f5 f6 eval', '')")
tail = tail.replace("submission_v8.csv')", "submission_v10.csv')").replace("'./v8_work'", "'./v10_work'")
assert 'feat_sq10' in tail and 'seq10.py f3' in tail
open('/workspace/huawei/v10/sol/solution_v10.py', 'w').write(head + 'SRC = ' + repr(SRC) + tail)
print('ok', MS)
