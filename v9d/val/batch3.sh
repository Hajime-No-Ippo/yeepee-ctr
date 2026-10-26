cd /workspace/huawei/v9d/val; P=/workspace/venv/bin/python; LR=feat_lr2,feat_tt,feat_pp
export POLARS_MAX_THREADS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
t=$(date +%s); (cd /workspace/huawei/v8b/work && SQ_THREADS=4 SQ_OUT=/workspace/huawei/v9d/feat_sq taskset -c 0-3 $P seq.py eval 2>&1 | grep -v Warn | tail -3); echo "TIME sq27 $(( $(date +%s)-t ))"
LP='"max_bin":63,"force_col_wise":true'; SLOW='"learning_rate":0.03,"feature_fraction":0.55,"num_leaves":95,"lambda_l2":30'
t=$(date +%s); taskset -c 0-3 $P ens.py 27 ctl_b --seeds 0 --threads 4 --lr $LR --negfrac 0.5 --nfolds 4 --params "{$LP}" 2>&1 | grep -v Warn | tail -1
taskset -c 0-3 $P ens.py 27 ctl_s --seeds 0 --rounds 500 --threads 4 --lr $LR --negfrac 0.5 --nfolds 4 --params "{$SLOW,$LP}" 2>&1 | grep -v Warn | tail -1; echo "TIME ctl27 $(( $(date +%s)-t ))"
t=$(date +%s); taskset -c 0-3 $P lr.py 27 0.1 2>&1 | grep -v Warn | tail -1; echo "TIME lr27 $(( $(date +%s)-t ))"
echo BATCH3 DONE
