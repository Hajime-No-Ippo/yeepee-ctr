#!/bin/bash
# ship.sh AUC25 RW : gate + candidate zip + sha
cd /workspace/huawei/v11
python3 /workspace/huawei/tools/package_final.py --csv /workspace/huawei/v11/out/submission_v11.csv --auc25 $1 --rw25 $2 --name v11; echo gate_exit=$?
cd out; rm -f submission_v11_candidate.csv.zip; /workspace/venv/bin/python -c "
import zipfile
z=zipfile.ZipFile('submission_v11_candidate.csv.zip','w',zipfile.ZIP_DEFLATED); z.write('submission_v11.csv',arcname='submission_v11.csv')"
sha256sum submission_v11_candidate.csv.zip submission_v11.csv | tee submission_v11_candidate.csv.zip.sha256
