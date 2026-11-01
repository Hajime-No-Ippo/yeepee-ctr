while pgrep -f "seq21.py 25 v21seq_big --var A" >/dev/null; do sleep 20; done
./run.sh 4-7 25 ctl6 A 0,1,2,3,4,5,6,7 --ep 6 --noseq --alpha 0.3
./run.sh 4-7 26 v21seq_big A 0,1 --ep 6 --elr 0 --enc trf --aug 0.3 --hd 128 --tok 256 --alpha 0.3
./run.sh 4-7 27 v21seq_big A 0,1,2 --ep 6 --elr 0 --enc trf --aug 0.3 --hd 128 --tok 256 --alpha 0.3
./run.sh 4-7 25 ctl6 B 0,1,2,3,4,5,6,7 --ep 6 --noseq --alpha 0.3
./run.sh 4-7 26 ctl6 B 0,1,2,3,4,5,6,7 --ep 6 --noseq --alpha 0.3
