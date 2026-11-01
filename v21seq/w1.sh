while pgrep -f "seq21.py 27 ctl6 --var A" >/dev/null; do sleep 20; done
./run.sh 0-3 27 ctl6 A 0,1,2,3,4,5,6,7 --ep 6 --noseq --alpha 0.3
./run.sh 0-3 26 ctl6 A 0,1,2,3,4,5,6,7 --ep 6 --noseq --alpha 0.3
./run.sh 0-3 23 ctl6 A 0,1,2,3,4,5,6,7 --ep 6 --noseq --alpha 0.3
./run.sh 0-3 27 ctl6 B 0,1,2,3,4,5,6,7 --ep 6 --noseq --alpha 0.3
./run.sh 0-3 23 ctl6 B 0,1,2,3,4,5,6,7 --ep 6 --noseq --alpha 0.3
