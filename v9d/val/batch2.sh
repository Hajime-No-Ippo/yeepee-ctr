cd /workspace/huawei/v9d/val; LR=feat_lr2,feat_tt,feat_pp; P=/workspace/venv/bin/python
export POLARS_MAX_THREADS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
t=$(date +%s); ./scr.sh nf5 $LR "26" --nfolds 5; echo "TIME nf5 $(( $(date +%s)-t ))"; $P cmp.py nf5
t=$(date +%s); ./scr.sh neg1 $LR "25 26 23" --negfrac 1.0; echo "TIME neg1 $(( $(date +%s)-t ))"; $P cmp.py neg1
for c in 25 26 23; do t=$(date +%s); taskset -c 0-3 $P lr.py $c 0.1 2>&1 | grep -v Warn | tail -2; echo "TIME lr$c $(( $(date +%s)-t ))"; done
echo BATCH2 DONE
