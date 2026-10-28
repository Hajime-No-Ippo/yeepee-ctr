#!/bin/bash
cd /workspace/huawei/v16seq
for spec in gru_s0 gru_s0+gru_s1+gru_s2 sas_s0 gru_s0+gru_s1+gru_s2,sas_s0; do
  echo "== $spec $(date)" 
  /workspace/venv/bin/python resid16.py oof $spec 0
done
echo R1_DONE
