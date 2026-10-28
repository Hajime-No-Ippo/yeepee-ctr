# v10: start the sequence-model stage in the background (it only needs v3/data from prep_v3) pinned to the last core with 1 torch thread, while the polars feature stages run on the other cores
rm -f seq10.rc seq10.log
( taskset -c "$2" "$1" seq10.py f3 f4 f5 f6 eval > seq10.log 2>&1; echo $? > seq10.rc ) < /dev/null > /dev/null 2>&1 &
echo "seq10 started in background on core $2 (SQ_THREADS=$SQ_THREADS)"
