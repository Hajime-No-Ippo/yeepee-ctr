#!/bin/bash
cd /workspace/huawei/v12; export POLARS_MAX_THREADS=8 OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
run(){ t0=$(date +%s); /workspace/venv/bin/python /workspace/huawei/v12x/xlgb.py "$@" 2>&1 | grep -v Warn | tail -2
  echo "{\"args\":\"$*\",\"wall_s\":$(( $(date +%s)-t0 ))}" >> /workspace/huawei/v12x/ens/timings.jsonl; }
LR='{"objective":"lambdarank","lambdarank_truncation_level":10,"metric":"None"}'
XE='{"objective":"rank_xendcg","metric":"None"}'
run 23 lrank --group --rounds 1000 --save 500,1000 --params "$LR"
run 25 xendcg --group --rounds 1000 --save 500,1000 --params "$XE"
run 23 xendcg --group --rounds 1000 --save 500,1000 --params "$XE"
run 25 mb1023 --rounds 1000 --save 1000 --params '{"max_bin":1023}'
run 23 mb1023 --rounds 1000 --save 1000 --params '{"max_bin":1023}'
run 25 mb511 --rounds 1000 --save 1000 --params '{"max_bin":511}'
run 23 mb511 --rounds 1000 --save 1000 --params '{"max_bin":511}'
echo Q4_DONE
