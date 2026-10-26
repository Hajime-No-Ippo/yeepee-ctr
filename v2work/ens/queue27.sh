P=/workspace/venv/bin/python
set -e
date +%T; $P rcov.py E1 E2 E3 E4 E5 E6 eval
date +%T; $P ens.py 27 base --seeds 0,1,2,3,4,5,6 --threads 4
date +%T; $P ens.py 27 wide --seeds 0,1 --threads 4 --params '{"feature_fraction":0.4,"num_leaves":127,"min_data_in_leaf":400}'
date +%T; $P ens.py 27 slow --seeds 0 --rounds 500 --threads 4 --params '{"learning_rate":0.03,"feature_fraction":0.55,"num_leaves":95,"lambda_l2":30}'
date +%T; $P ens.py 27 estrc --seeds 0,1 --folds e --stale --rc --threads 4
date +%T; echo Q27DONE
