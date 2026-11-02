import sys; sys.path.insert(0,'/workspace/huawei/cb_blend/evclick')
from evlib import *
import time
r=json.load(open(H+'/v20/gain_rank.json'))['ranked']; top=[x['feature'] for x in r[:70]]
f484=json.load(open(H+'/v20/feats_ref484.json'))
USR=[c for c in f484 if c.startswith('us_')]+['v2_u_ctr']
HIST=[c for c in f484 if c.startswith('hist_')]
ROWF=[c for c in top if c not in USR and c not in HIST]; LSEF=[c for c in ROWF if re.search(r'dot|logit|cos',c)]
json.dump(dict(ROWF=ROWF,USR=USR,HIST=HIST,LSEF=LSEF),open('evfeats.json','w'))
def build(n):
    t0=time.time(); df,miss=cols(n,ROWF+USR+HIST); assert not miss,miss
    df=df.with_columns([pl.col(c).cast(pl.Float32) for c in ROWF+USR+HIST])
    aggs=[]
    for c in ROWF:
        aggs+=[pl.col(c).mean().alias(f'{c}__mean'),pl.col(c).max().alias(f'{c}__max'),pl.col(c).min().alias(f'{c}__min'),pl.col(c).std().alias(f'{c}__std')]
    for c in LSEF:
        aggs+=[pl.col(c).sort(descending=True).slice(1,1).first().alias(f'{c}__2nd'),(pl.col(c)*1.0).exp().sum().log().alias(f'{c}__lse')]
    for c in HIST: aggs+=[pl.col(c).mean().alias(f'{c}__mean')]
    aggs+=[pl.col(c).first() for c in list(dict.fromkeys(USR+['tm_stale','us_cold','us_nohist']))]
    aggs+=[pl.col('u').first(),pl.col('ts').first(),pl.col('rowid').first().alias('rowid0'),pl.col('clicked').sum().alias('k') if 'clicked' in df.columns else pl.lit(0).alias('k'),pl.len().alias('n')]
    ev=df.group_by('eid',maintain_order=True).agg(aggs)
    ev.write_parquet(f'ev_{n}.parquet'); print(n,ev.shape,round(time.time()-t0),flush=True)
    df.select(['rowid','eid']).write_parquet(f'rows_{n}.parquet')
    del df,ev; gc.collect()
for n in ['f1','f2','f3','f4','f5','f6','val23','val25','val26','eval']: build(n)
