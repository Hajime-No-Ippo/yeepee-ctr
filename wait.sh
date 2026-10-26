#!/bin/bash
cd /workspace/huawei/parts
start=$(date +%s); last_change=$(date +%s); lastsig=""
while true; do
  now=$(date +%s)
  sig=$(ls -l /workspace/huawei | md5sum)$(ls -l | md5sum)
  if [ "$sig" != "$lastsig" ]; then last_change=$now; lastsig=$sig; fi
  lastpart=$(ls train.zip.part_* | sort | tail -1)
  eocd=$(python3 -c "
import sys;f=open('$lastpart','rb');f.seek(0,2);n=f.tell();f.seek(max(0,n-70000));print(int(b'PK\x05\x06' in f.read()))")
  echo "$(date +%T) parts=$(ls train.zip.part_* | wc -l) last=$lastpart size=$(stat -c %s $lastpart) eocd=$eocd idle=$((now-last_change))s"
  ls /workspace/huawei
  if [ "$eocd" = 1 ] && [ $((now-last_change)) -ge 180 ]; then echo DONE; exit 0; fi
  if [ $((now-start)) -ge 5400 ]; then echo TIMEOUT; exit 1; fi
  sleep 60
done
