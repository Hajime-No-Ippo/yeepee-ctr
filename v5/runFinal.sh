#!/bin/bash
cd /workspace/huawei/v5; source env.sh
until grep -q A4_DONE out/runA4.out 2>/dev/null; do sleep 15; done
$PY decide.py vA > out/decide.log 2>&1; rc=$?
[ $rc -ne 0 ] && { echo "DECISION_NOT_PASSED rc=$rc"; echo FINAL_DONE; exit 0; }
./guard.sh; echo "=== $(date +%T) combo final"
[ -f eval_pred_v5.parquet ] || XF=1 XF2=1 XFDIR=feat_xA OUTP=eval_pred_v5 nice -n 19 $PY combo.py final 400 3 notime > out/final_v5.log 2>&1
echo "=== $(date +%T) write_sub"
PRED=/workspace/huawei/v5/eval_pred_v5.parquet $PY write_sub.py /workspace/huawei/v5/submission_v5.csv > out/write_sub_v5.log 2>&1
$PY validate.py /workspace/huawei/v5/submission_v5.csv > out/validate_v5.log 2>&1; echo "validate rc=$?" >> out/validate_v5.log
$PY /workspace/huawei/v3/validate_submission.py /workspace/huawei/v5/submission_v5.csv > out/validate_v3_v5.log 2>&1; echo "v3 validator rc=$?" >> out/validate_v3_v5.log
$PY -c "
import polars as pl; from scipy.stats import spearmanr
a=pl.read_csv('/workspace/huawei/v5/submission_v5.csv',schema_overrides={'rowid':pl.Utf8}); b=pl.read_csv('/workspace/huawei/v2work/out/submission_v3.csv',schema_overrides={'rowid':pl.Utf8})
j=a.join(b,on='rowid',suffix='_v3'); print('spearman_vs_v3',round(spearmanr(j['predicted_probability'],j['predicted_probability_v3']).correlation,5),'rows',j.height)" > out/spearman_v5.log 2>&1
echo FINAL_DONE
