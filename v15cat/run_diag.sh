#!/bin/bash
cd /workspace/huawei/v12; export POLARS_MAX_THREADS=8 PYTHONUNBUFFERED=1
P=/workspace/venv/bin/python; C=/workspace/huawei/v15cat/catrun.py; L=/workspace/huawei/v15cat/logs
$P $C 25 catA_noid --set A --dropcat '^(item_id|user_id)$' > $L/catA_noid_c25.log 2>&1
$P $C 25 catA_nocat --set A --dropcat '.' > $L/catA_nocat_c25.log 2>&1
$P $C 25 catB --set B > $L/catB_c25.log 2>&1
echo DIAG_DONE >> $L/diag.log
