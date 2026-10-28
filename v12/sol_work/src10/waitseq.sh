# v10: block until the background sequence stage finishes; fail if it failed
while [ ! -f seq10.rc ]; do sleep 1; done
grep -v ' ep' seq10.log; exit $(cat seq10.rc)
