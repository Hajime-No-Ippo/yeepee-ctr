import sys,os,glob,json,re
sys.path.insert(0,'/workspace/huawei/v20')
import polars as pl, lib20
from cl import matrix
X,m,f=matrix('val25',1.0,variant='notime',extra=lib20.EXTRA)
print('matrix cols before DROP',len(f)); d=[c for c in f if re.search(lib20.DROP,c)]; print('dropped',d)
ref=json.load(open('/workspace/huawei/v20/feats_ref484.json')); print('ref484',len(ref),'in f:',len(set(ref)&set(f)),'f not in ref',[c for c in f if c not in ref][:20])
json.dump(f,open('/workspace/huawei/research/blending_v1_scripts/out/matrix_cols_val25.json','w'))
for base in ['v12','v12x','v17cooc','v19','v21trend','v21cat','v21seq','v21graph','v21dcn','v21roll']:
    for p in sorted(glob.glob(f'{base}/feat_*/val25.parquet')+glob.glob(f'{base}/*/*/val25.parquet')):
        try: n=len(pl.scan_parquet(p).collect_schema().names())-1
        except Exception as e: n=str(e)[:30]
        print(p,n)
