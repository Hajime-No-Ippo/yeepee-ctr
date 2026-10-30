"""Select the 25 co-occurrence columns used by v17cooc: feat_{A,B}/*.parquet (58 cols from mkcooc.py) -> feat_{A,B}p/*.parquet. Run with cwd = work dir."""
import polars as pl, os, glob, sys
for v in (sys.argv[1] if len(sys.argv)>1 else 'AB'):
    os.makedirs(f'feat_{v}p',exist_ok=True)
    for f in sorted(glob.glob(f'feat_{v}/*.parquet')):
        d=pl.read_parquet(f); n=os.path.basename(f)
        cols=[c for c in d.columns if c!='rowid' and c[4:].startswith(('pp12_cos','pp8','pp16','pp12x','ppb12','pp12_unorm','pp12_ucos','pp_prod'))]
        d.select(['rowid']+cols).write_parquet(f'feat_{v}p/{n}')
    print(v,len(cols),cols)
