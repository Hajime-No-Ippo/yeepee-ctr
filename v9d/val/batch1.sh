cd /workspace/huawei/v9d/val; LR=feat_lr2,feat_tt,feat_pp
t=$(date +%s); ./scr.sh wadv $LR "25 26 23" --wtag advni --wpow 0.5; echo "TIME wadv $(( $(date +%s)-t ))"; /workspace/venv/bin/python cmp.py wadv
t=$(date +%s); ./scr.sh nf3 $LR "25 26" --nfolds 3; echo "TIME nf3 $(( $(date +%s)-t ))"; /workspace/venv/bin/python cmp.py nf3
t=$(date +%s); ./scr.sh nf2 $LR "25 26" --nfolds 2; echo "TIME nf2 $(( $(date +%s)-t ))"
t=$(date +%s); ./scr.sh nf1 $LR "23" --nfolds 1; echo "TIME nf1c23 $(( $(date +%s)-t ))"; /workspace/venv/bin/python cmp.py nf2 nf1
echo BATCH1 DONE
