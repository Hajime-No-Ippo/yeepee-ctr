#!/bin/bash
cd /workspace/huawei/v16seq; export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
P=/workspace/venv/bin/python
for s in 0 1 2; do
 $P seqfeat.py gru $s C3 val23 > logs/gru_s${s}_C3.log 2>&1; $P seqfeat.py gru $s C5 val25 > logs/gru_s${s}_C5.log 2>&1; $P seqfeat.py gru $s C6 val26 > logs/gru_s${s}_C6.log 2>&1
done
$P seqfeat.py sas 0 C3 val23 > logs/sas_s0_C3.log 2>&1; $P seqfeat.py sas 0 C5 val25 > logs/sas_s0_C5.log 2>&1; $P seqfeat.py sas 0 C6 val26 > logs/sas_s0_C6.log 2>&1
echo S1_DONE
