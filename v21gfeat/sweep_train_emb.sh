#!/bin/bash
# train-only LightGCN embeddings for the earlier cutoffs k=1,2,4 (needed so that f1..f6 training rows get graph feats from graph BEFORE their own cutoff)
# writes into v21gfeat/out (own copy of gcn.py outputs); reads graphs/slates from v21graph/out (read-only)
cd /workspace/huawei/v21gfeat; source /workspace/venv_v17/bin/activate
for s in 0 1 2 3 4 5 9; do
 for k in 4 2 1; do
  if [ $k = 4 ]; then TAG=u1; ARGS="--steps 350"; ST=350; else TAG=u1b; ARGS="--steps 800"; ST=800; fi
  [ -f out/emb_${TAG}_k${k}_s${s}.npz ] && continue
  taskset -c 4-7 python -u gcn_k.py $k $TAG --seed $s $ARGS --lr 1e-2 --L 3 --d 64 --l2 3e-3 --save $ST --thr 2 > logs/emb_${TAG}_k${k}_s${s}.log 2>&1
 done
done
echo ALLDONE > logs/emb_done.txt
