"""build /workspace/huawei/v12/sol/solution_v12.py from sol_work/src12 (+ head12/tail12)."""
import os
B = '/workspace/huawei/v12/sol_work'; S = f'{B}/src12'; SRC = {}
for r, _, fs in os.walk(S):
    for f in sorted(fs):
        if f.endswith(('.py', '.sh')) and '__pycache__' not in r:
            p = os.path.join(r, f); SRC[os.path.relpath(p, S)] = open(p).read()
head = open(f'{B}/head12.txt').read(); tail = open(f'{B}/tail12.txt').read()
open('/workspace/huawei/v12/sol/solution_v12.py', 'w').write(head + 'SRC = ' + repr(dict(sorted(SRC.items()))) + tail)
print('ok', sorted(SRC))
