"""Validate a submission against evaluation rowids. Usage: python validate_submission.py SUB.csv [EXAMPLE.csv]"""
import sys, math, csv
sub = sys.argv[1]; ex = sys.argv[2] if len(sys.argv) > 2 else '/workspace/huawei/example_submission.csv'
errs = []
with open(ex, newline='', encoding='utf-8') as f:
    r = csv.reader(f); next(r); ex_ids = [row[0] for row in r]
with open(sub, newline='', encoding='utf-8') as f:
    r = csv.reader(f); header = next(r); rows = list(r)
if header != ['rowid', 'predicted_probability']: errs.append(f'bad header {header}')
ids = [row[0] for row in rows]
if any(len(row) != 2 for row in rows): errs.append('rows with != 2 columns')
if len(rows) != len(ex_ids): errs.append(f'row count {len(rows)} != {len(ex_ids)}')
if len(set(ids)) != len(ids): errs.append(f'{len(ids)-len(set(ids))} duplicate rowids')
s_ex = set(ex_ids); s_sub = set(ids)
if s_sub - s_ex: errs.append(f'{len(s_sub-s_ex)} unknown rowids')
if s_ex - s_sub: errs.append(f'{len(s_ex-s_sub)} missing rowids')
same_order = ids == ex_ids
bad = 0; vals = []
for row in rows:
    try:
        v = float(row[1])
        if not math.isfinite(v) or v < 0 or v > 1: bad += 1
        else: vals.append(v)
    except Exception:
        bad += 1
if bad: errs.append(f'{bad} invalid values (non-numeric/NaN/inf/outside [0,1])')
print(f'file={sub} rows={len(rows)} unique={len(s_sub)} same_order_as_example={same_order}')
if vals:
    vs = sorted(vals); n = len(vs)
    print(f'min={vs[0]:.6g} p50={vs[n//2]:.6g} max={vs[-1]:.6g} mean={sum(vs)/n:.6g} distinct={len(set(vs))}')
print('VALID' if not errs else 'INVALID: ' + '; '.join(errs))
sys.exit(1 if errs else 0)
