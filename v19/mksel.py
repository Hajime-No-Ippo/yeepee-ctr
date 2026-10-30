"""select pruned NMF set: all nmfP (15) + nmfU scalar (cos,cos_z,cos_dm,aff,aff_z,lift,cos_e10,wu_rec,wu_rec_z); drop LDA and nmfU per-dim products."""
import polars as pl, os, glob
for v in 'AB':
    os.makedirs(f'feat_N{v}s',exist_ok=True)
    for f in sorted(glob.glob(f'feat_N{v}/*.parquet')):
        d=pl.read_parquet(f); n=os.path.basename(f); p=f'nm{v}_'
        cols=[c for c in d.columns if c.startswith(p+'nmfP_') or c in [p+'nmfU_'+x for x in ('cos','cos_z','cos_dm','aff','aff_z','lift','cos_e10','wu_rec','wu_rec_z')]]
        d.select(['rowid']+cols).write_parquet(f'feat_N{v}s/{n}')
    print(v,len(cols))
