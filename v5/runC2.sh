#!/bin/bash
cd /workspace/huawei/v5; source env.sh
until grep -q C_DONE out/runC.out 2>/dev/null; do sleep 15; done
$PY - > out/c1_screen_gate.log <<'P'
import json
R={}
for l in open('out/results.jsonl'):
    try:r=json.loads(l)
    except:continue
    if r.get('kind')=='screen': R[(r['tag'],r['cut'])]=r
a=lambda r:r['curve'][max(r['curve'],key=int)]
try:
    d25=a(R[('scr_C1',25)])-a(R[('scr_Astack',25)]); d26=a(R[('scr_C1',26)])-a(R[('scr_Astack',26)])
    print('C1 vs Astack d25=%+.5f d26=%+.5f'%(d25,d26)); print('GATE','PASS' if d25>=0.0005 and d26>=0 else 'FAIL')
except KeyError as e: print('missing',e); print('GATE FAIL')
P
cat out/c1_screen_gate.log
grep -q "GATE PASS" out/c1_screen_gate.log || { echo C2_DONE; exit 0; }
./guard.sh; echo "=== $(date +%T) confirm vC"
grep -q '"tag": "vC", "step": "C1", "cut": 23' out/results.jsonl || XDIRS=feat_xA,feat_x2,feat_pp nice -n 19 $PY v5val.py vC C1 25 26 23 > out/vC.log 2>&1
$PY decide_vs.py vC vA > out/decide_vC.log 2>&1 || { cat out/decide_vC.log; echo C2_DONE; exit 0; }
cat out/decide_vC.log
./guard.sh; echo "=== $(date +%T) combo final v5c"
[ -f eval_pred_v5c.parquet ] || XF=1 XF2=1 XFDIR=feat_xA XF3=feat_pp OUTP=eval_pred_v5c nice -n 19 $PY combo.py final 400 3 notime > out/final_v5c.log 2>&1
PRED=/workspace/huawei/v5/eval_pred_v5c.parquet $PY write_sub.py /workspace/huawei/v5/submission_v5c.csv > out/write_sub_v5c.log 2>&1
$PY validate.py /workspace/huawei/v5/submission_v5c.csv > out/validate_v5c.log 2>&1; echo "rc=$?" >> out/validate_v5c.log
$PY /workspace/huawei/v3/validate_submission.py /workspace/huawei/v5/submission_v5c.csv > out/validate_v3_v5c.log 2>&1; echo "rc=$?" >> out/validate_v3_v5c.log
$PY -c "
import polars as pl; from scipy.stats import spearmanr
c=pl.read_csv('/workspace/huawei/v5/submission_v5c.csv',schema_overrides={'rowid':pl.Utf8})
for p in ['/workspace/huawei/v5/submission_v5.csv','/workspace/huawei/v2work/out/submission_v3.csv']:
    b=pl.read_csv(p,schema_overrides={'rowid':pl.Utf8}); j=c.join(b,on='rowid',suffix='_o'); print('spearman',p,round(spearmanr(j['predicted_probability'],j['predicted_probability_o']).correlation,5))" > out/spearman_v5c.log 2>&1
echo C2_DONE
