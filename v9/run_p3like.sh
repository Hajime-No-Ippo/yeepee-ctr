cd /workspace/huawei/v9; P=/workspace/venv/bin/python
export POLARS_MAX_THREADS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
LP='"max_bin":63,"force_col_wise":true'
for c in 25 23; do taskset -c 0-3 $P ens.py $c p3like_b --seeds 0 --threads 4 --lr feat_lr2,feat_tt,feat_pp --negfrac 0.5 --nfolds 4 --keep keep_p3like.txt --params "{$LP}" 2>&1 | grep -v Warn | tail -2; done
echo DONE
