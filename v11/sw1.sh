#!/bin/bash
# raw seq sweep on val25 (single J25 fit). baseline v10 cfg raw C25 .78236
cd /workspace/huawei/v11; export PYTHONUNBUFFERED=1 SQ_WORK=/workspace/huawei SQ_THREADS=8
while ! grep -q B1_DONE logs/b1.log; do sleep 20; done
run(){ n=$1; shift; t0=$(date +%s); env "$@" SQ_OUT=sw/$n /workspace/venv/bin/python seq11.py val25 2>&1 | grep -v " ep[0-9]" | tail -2; echo "$n wall $(( $(date +%s)-t0 ))s $*"; /workspace/venv/bin/python sqauc.py sw/$n; }
run base SQ_X=0
run ep5 SQ_EP=5 SQ_MAXSTEPS=100000
run d64 SQ_D=64 SQ_DH=32
run temb SQ_TEMB=1
run hl50 SQ_HL=50
run np6 SQ_NP=6
run h4 SQ_H=4 SQ_HID=128
run seed1 SQ_SEED=1
echo SW1_DONE
