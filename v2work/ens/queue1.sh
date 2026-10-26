P=/workspace/venv/bin/python
for c in 25 26; do
 if [ $c = 26 ]; then $P ens.py 26 base --seeds 0,1,2,3,4,5,6; else $P ens.py 25 base --seeds 1,2,3,4,5,6; fi
 $P ens.py $c wide --seeds 0,1 --params '{"feature_fraction":0.4,"num_leaves":127,"min_data_in_leaf":400}'
 $P ens.py $c slow --seeds 0 --rounds 500 --params '{"learning_rate":0.03,"feature_fraction":0.55,"num_leaves":95,"lambda_l2":30}'
 $P ens.py $c nov2 --seeds 0 --drop '^v2_'
 $P ens.py $c estrc --seeds 0,1 --folds e --stale --rc
done
echo Q1DONE
